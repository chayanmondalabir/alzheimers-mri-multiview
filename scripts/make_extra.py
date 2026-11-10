# Run once from the repository root:  python scripts/make_extra.py path/to/qc.csv
# qc.csv comes from the OASIS-2 preprocessing notebook (it is not stored in this repo).
import sys, json, pandas as pd
qc = pd.read_csv(sys.argv[1])
json.dump(dict(target_vox=float(qc.brain_voxels.median()), ncc_p20=float(qc.ncc_to_template.quantile(0.2))),
          open("weights/extra.json", "w"), indent=1)
print(open("weights/extra.json").read())
