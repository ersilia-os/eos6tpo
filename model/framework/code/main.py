# imports
import os
import csv
import sys
import json
import subprocess
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))


# run model
cmd = [
        sys.executable, "-m", "chebifier", "predict",
        "--smiles-file", os.path.join(root, "..", input_file),
        "--output", os.path.join(root, "..", output_file.replace(".csv", ".json")),
        "--ensemble-config", os.path.join(root, "..", "..", "checkpoints", "ensemble_config.yml"),
    ]
subprocess.run(cmd, check=True, cwd=os.path.join(root, "python-chebifier"))

# read input smiles from .csv file
smiles = [i.strip() for i in open(os.path.join(root, "..", input_file), "r").readlines()[1:]]

# read json output
output = json.load(open(os.path.join(root, "..", output_file.replace(".csv", ".json"))))
output_content = ['chebi_predicted_parents']
for smi in smiles:
    r = ["CHEBI:" + o for o in sorted(output[smi])]
    r = ";".join(r)
    output_content.append(r)

# remove json output
os.remove(os.path.join(root, "..", output_file.replace(".csv", ".json")))

# write output in a .csv file
with open(os.path.join(root, "..", output_file), "w") as f:
    f.write("\n".join(output_content))
