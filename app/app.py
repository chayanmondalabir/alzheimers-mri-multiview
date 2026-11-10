import os, tempfile
import numpy as np, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inference as inf

st.set_page_config(page_title="MRI dementia screening demo", layout="wide")
st.title("Age-Controlled Alzheimer's Disease Detection from Structural MRI")
st.caption("Multi-view 2.5D ResNet-18 ensemble · trained on OASIS-1 (ages 60+) · research demo of a conference paper")
st.warning("**Research prototype. Not a medical device and not for diagnosis.** "
           "The model was trained and tested only on OASIS data (AUC 0.751 in cross-validation, 0.852 on OASIS-2). "
           "Very mild dementia (CDR 0.5) is often missed.")

up = st.file_uploader("Upload a T1-weighted brain MRI (.nii or .nii.gz). Ages 60+ only; standard orientation.", type=["nii", "gz", "npz"])
st.caption("Uploaded files are processed in memory/temp storage and deleted right after the run. Do not upload data that identifies a person.")

if up is not None and st.button("Run analysis", type="primary"):
    suffix = ".npz" if up.name.endswith(".npz") else (".nii.gz" if up.name.endswith(".gz") else ".nii")
    tmp = os.path.join(tempfile.mkdtemp(), "upload" + suffix)
    open(tmp, "wb").write(up.getbuffer())
    status = st.empty()
    try:
        if suffix == ".npz":
            vol = np.load(tmp)["vol"].astype(np.float32)          # already-prepared volume (testing mode)
        else:
            vol = inf.preprocess_raw(tmp, progress=lambda s: status.info(s + " ... (this can take a few minutes on CPU)"))
        status.info("Running the three networks")
        p, per_view = inf.predict(vol)
        ncc = inf.ncc_to_template(vol)
        status.empty()
        c1, c2, c3 = st.columns(3)
        c1.metric("Dementia probability (ensemble)", f"{p:.2f}")
        c2.metric("Result at threshold 0.5", "Dementia-like (CDR > 0)" if p >= 0.5 else "Control-like (CDR 0)")
        c3.metric("Alignment quality", f"{ncc:.2f}")
        st.write("Probability per plane: " + " · ".join(f"{v} {x:.2f}" for v, x in per_view.items()))
        if ncc < inf.QC_P20:
            st.warning("Alignment with the template is in the weakest 20% of the OASIS-2 scans in the paper. Treat this result with extra caution.")
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        for a, v in zip(ax, inf.VIEWS):
            bg, cam = inf.grad_cam(vol, v)
            a.imshow(bg.T, cmap="gray", origin="lower"); a.imshow(cam.T, cmap="jet", alpha=np.clip(cam.T, 0, 1) * 0.6, vmin=0, vmax=1, origin="lower")
            a.set_title(v); a.axis("off")
        st.subheader("Where the network looked (Grad-CAM, central slice)")
        st.pyplot(fig)
        st.caption("In the paper these maps were only partly supported by faithfulness tests, so they are supporting evidence, not proof.")
    except Exception as e:
        status.empty(); st.error(f"Could not process this file: {e}")
    finally:
        try: os.remove(tmp)
        except OSError: pass
