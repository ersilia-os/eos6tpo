# imports
import os
import sys
import json

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# import chebifier
sys.path.insert(0, os.path.join(root, "python-chebifier"))
from chebifier.cli_adapted import predict

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# change working directory before running the model
os.chdir(os.path.join(root, "python-chebifier"))

# run the model
predict(
    ensemble_config=os.path.join(root, "..", "..", "checkpoints", "ensemble_config.yml"),
    smiles=(),  # none inline
    smiles_file=os.path.join(root, "..", input_file),
    output=os.path.join(root, "..", output_file.replace(".csv", ".json")),
    ensemble_type="wmv-f1",
    chebi_version=241,
    use_confidence=True,
    resolve_inconsistencies=True
)


# # run model
# cmd = [
#         sys.executable, "-m", "chebifier", "predict",
#         "--smiles-file", os.path.join(root, "..", input_file),
#         "--output", os.path.join(root, "..", output_file.replace(".csv", ".json")),
#         "--ensemble-config", os.path.join(root, "..", "..", "checkpoints", "ensemble_config.yml"),
#     ]
# subprocess.run(cmd, check=True, cwd=os.path.join(root, "python-chebifier"))

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
