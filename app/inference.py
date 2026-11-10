"""Inference code for the multi-view 2.5D ResNet-18 ensemble (research demo).
Preprocessing copies the OASIS-2 pipeline of the paper (final version used for the paper's external test):
RAS reorientation -> N4 -> ANTsPyNet brain extraction (largest component) -> scale to typical OASIS-1 brain size
-> centre on the OASIS-1 template -> clip 1/99.5 percentile + z-score. No flips, no registration."""
import os, glob, json, math, tempfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.environ.get("WEIGHTS_DIR", os.path.join(HERE, "..", "weights"))
CFG = json.load(open(os.path.join(W, "config.json")))
VIEWS = CFG["views"]; VIEW_AXIS = CFG["view_axis"]
TEMPLATE = np.load(os.path.join(W, "template.npy")).astype(np.float32)
# typical OASIS-1 brain size (voxels). Taken from the prepared OASIS-2 volumes (they were scaled to this size).
TARGET_VOX = float(json.load(open(os.path.join(W, "extra.json")))["target_vox"])
QC_P20 = float(json.load(open(os.path.join(W, "extra.json")))["ncc_p20"])
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ------------------------------------------------------------------ preprocessing
def _com(a): return np.argwhere(a != 0).mean(0)

def _largest(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1: return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))

def normalise(vol):
    m = vol > 0
    if m.sum() <= 1000: raise ValueError("Empty brain after skull-stripping. Is this a T1 head MRI?")
    v = vol[m]; lo, hi = np.percentile(v, [1, 99.5]); v = np.clip(v, lo, hi)
    mu, sd = v.mean(), v.std() + 1e-6
    out = np.zeros_like(vol, dtype=np.float32)
    out[m] = (np.clip(vol[m], lo, hi) - mu) / sd
    return out

def preprocess_raw(path, progress=lambda s: None):
    import nibabel as nib, ants, antspynet
    tmp = os.path.join(tempfile.mkdtemp(), "in.nii.gz")
    progress("Reading and reorienting the scan")
    im = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(im.dataobj).astype(np.float32)
    while data.ndim > 3: data = data[..., 0]
    nib.save(nib.Nifti1Image(data, im.affine), tmp)
    img = ants.image_read(tmp)
    progress("Bias-field correction (N4)")
    img = ants.n4_bias_field_correction(img)
    progress("Skull-stripping (ANTsPyNet)")
    prob = antspynet.brain_extraction(img, modality="t1")
    mask = _largest(prob.numpy() > 0.5).astype("float32")
    brain = img.numpy() * mask
    sp = np.array(img.spacing)
    progress("Scaling and centring on the template")
    fixed = ants.from_numpy(TEMPLATE, spacing=(1.0, 1.0, 1.0)); com_fixed = _com(TEMPLATE) * 1.0
    def frame(arr, spc):
        origin = tuple(com_fixed - _com(arr) * np.array(spc))
        return ants.from_numpy(np.ascontiguousarray(arr.astype("float32")), origin=origin, spacing=tuple(float(x) for x in spc))
    s = (TARGET_VOX / max(float((mask > 0).sum() * np.prod(sp)), 1.0)) ** (1 / 3.0)
    sp2 = sp * s
    mov, movm = frame(brain, sp2), frame(mask, sp2)
    rough = ants.resample_image_to_target(mov, fixed, interp_type=0).numpy()
    roughm = ants.resample_image_to_target(movm, fixed, interp_type=1).numpy() > 0.5
    vol = np.where(rough * roughm > 0, rough * roughm, 0)
    try: os.remove(tmp)
    except OSError: pass
    return normalise(vol)

def ncc_to_template(v):
    m = (v != 0) | (TEMPLATE != 0)
    x, y = v[m] - v[m].mean(), TEMPLATE[m] - TEMPLATE[m].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))

# ------------------------------------------------------------------ slices (same as the training notebook)
def view_slices(vol, axis, n, frac, step):
    nz = vol != 0
    other = tuple(a for a in range(3) if a != axis)
    idx = np.where(nz.sum(axis=other) > 50)[0]
    lo, hi = int(idx[0]), int(idx[-1])
    c, half = (lo + hi) / 2.0, frac * (hi - lo) / 2.0
    pos = np.linspace(c - half, c + half, n).round().astype(int)
    size = vol.shape[axis]; out = []
    for p in pos:
        ch = np.clip([p - step, p, p + step], 0, size - 1)
        out.append(np.moveaxis(np.take(vol, ch, axis=axis), axis, 0))
    return np.stack(out).astype(np.float16)

# ------------------------------------------------------------------ models
def make_model():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(512, 1)
    return m

_MODELS = None
def load_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = {}
        for v in VIEWS:
            lst = []
            for f in sorted(glob.glob(os.path.join(W, f"{v}_seed*.pt"))):
                m = make_model(); m.load_state_dict(torch.load(f, map_location="cpu")); m.eval(); lst.append(m)
            if not lst: raise FileNotFoundError(f"no weights for {v} in {W}")
            _MODELS[v] = lst
    return _MODELS

def _to_in(xb):
    return F.interpolate(xb.float(), size=(CFG["img"], CFG["img"]), mode="bilinear", align_corners=False)

@torch.no_grad()
def predict(vol):
    """returns (ensemble probability, {view: probability})"""
    models = load_models(); per_view = {}
    for v in VIEWS:
        sl = torch.from_numpy(view_slices(vol, VIEW_AXIS[v], CFG["n_slices"], CFG["central_frac"], CFG["step"]))
        x = _to_in(sl)
        ps = [torch.sigmoid(m(x).squeeze(1)).mean().item() for m in models[v]]   # mean over the 12 slices
        per_view[v] = float(np.mean(ps))                                         # mean over seeds
    return float(np.mean(list(per_view.values()))), per_view

def grad_cam(vol, view, slice_idx=None):
    """Grad-CAM on layer4 of the first seed's network for one plane. returns (background slice, cam) of the chosen slice."""
    m = load_models()[view][0]
    sl = torch.from_numpy(view_slices(vol, VIEW_AXIS[view], CFG["n_slices"], CFG["central_frac"], CFG["step"]))
    x = _to_in(sl); acts, grads = {}, {}
    h1 = m.layer4.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))
    h2 = m.layer4.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("g", go[0]))
    m.zero_grad(); out = m(x).squeeze(1); out.sum().backward(); h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].detach()
    cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-9)
    k = CFG["n_slices"] // 2 if slice_idx is None else slice_idx
    bg = x[k, 1].numpy(); c = cam[k].numpy(); c = np.where(np.abs(bg) > 1e-6, c, 0)
    return bg, c
