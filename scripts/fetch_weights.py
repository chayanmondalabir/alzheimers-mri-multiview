"""Download the trained weights (15 networks), the template and config.json from the GitHub Release.
Usage:  python scripts/fetch_weights.py            (uses the default URL below or $WEIGHTS_URL)
"""
import os, sys, zipfile, urllib.request

URL = os.environ.get("WEIGHTS_URL",
      "https://github.com/chayanmondalabir/alzheimers-mri-multiview/releases/download/v1.0/weights.zip")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")

def main():
    os.makedirs(DEST, exist_ok=True)
    if all(os.path.exists(os.path.join(DEST, f)) for f in ("config.json", "template.npy", "extra.json")):
        print("weights already present"); return
    zpath = os.path.join(DEST, "weights.zip")
    print("downloading", URL); urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z: z.extractall(DEST)
    os.remove(zpath); print("done:", sorted(os.listdir(DEST)))

if __name__ == "__main__":
    sys.exit(main())
