# imports
import os
import csv
import sys
import json
import yaml
import pickle

# current file directory (needed before patching)
root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))

# Patch chebifier to load chebi_graph.pkl and disjoint files from local checkpoints
# instead of HuggingFace. Must happen before cli_adapted is imported (which triggers
# base_ensemble import).
import chebifier.ensemble.base_ensemble as _base_ensemble
_orig_load_chebi_graph = _base_ensemble.load_chebi_graph
def _local_load_chebi_graph(filename=None):
    local = os.path.join(checkpoints_dir, "chebi_graph.pkl")
    if filename is None and os.path.isfile(local):
        print("Loading ChEBI graph from local checkpoints...")
        return pickle.load(open(local, "rb"))
    return _orig_load_chebi_graph(filename)
_base_ensemble.load_chebi_graph = _local_load_chebi_graph

_orig_get_disjoint_files = _base_ensemble.get_disjoint_files
def _local_get_disjoint_files():
    local = [
        os.path.join(checkpoints_dir, "disjoint_chebi.csv"),
        os.path.join(checkpoints_dir, "disjoint_additional.csv"),
    ]
    if all(os.path.isfile(f) for f in local):
        return local
    return _orig_get_disjoint_files()
_base_ensemble.get_disjoint_files = _local_get_disjoint_files

# Patch ChEBILookupPredictor.get_smiles_lookup to use an absolute path in checkpoints
# instead of a path relative to CWD (which breaks when CWD is /tmp).
from chebifier.prediction_models.chebi_lookup import ChEBILookupPredictor as _ChEBILookupPredictor

def _local_get_smiles_lookup(self_inner):
    local = os.path.join(checkpoints_dir, "smiles_lookup.json")
    if os.path.isfile(local):
        print("Loading SMILES lookup from local checkpoints...")
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    print("Building SMILES lookup (first run, may take a few minutes)...")
    smiles_lookup = self_inner.build_smiles_lookup()
    with open(local, "w", encoding="utf-8") as f:
        json.dump(smiles_lookup, f, indent=4)
    return smiles_lookup

_ChEBILookupPredictor.get_smiles_lookup = _local_get_smiles_lookup

# Patch ChEBIData to use local chebi.obo instead of downloading from the internet,
# and skip SDF-based molecular data processing (not needed for element classification).
import shutil
from chemlog.preprocessing.chebi_data import ChEBIData as _ChEBIData

_orig_download_chebi = _ChEBIData.download_chebi
def _local_download_chebi(self):
    chebi_path = getattr(self, 'chebi_path', None)
    if chebi_path is None:
        chebi_path = os.path.join("data", f"chebi_v{self.chebi_version}", "chebi.obo")
    if not os.path.isfile(chebi_path):
        local = os.path.join(checkpoints_dir, "chebi.obo")
        if os.path.isfile(local):
            os.makedirs(os.path.dirname(os.path.abspath(chebi_path)), exist_ok=True)
            shutil.copy2(local, chebi_path)
            print("Using local chebi.obo from checkpoints")
            return
    _orig_download_chebi(self)
_ChEBIData.download_chebi = _local_download_chebi

_orig_process_data = _ChEBIData.process_data
def _local_process_data(self):
    # by_element_classification only uses self.chebi (set by process_chebi),
    # not the SDF-derived molecular data. Skip if processed.pkl not already cached.
    processed_path = getattr(self, 'processed_path', None)
    if processed_path and os.path.isfile(str(processed_path)):
        return _orig_process_data(self)
    print("Skipping ChEBI molecular data processing (SDF not bundled, not needed for element classification)")
    return None
_ChEBIData.process_data = _local_process_data

from cli_adapted import predict

# parse arguments
input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])
tmp_file = os.path.abspath(output_file.replace(".csv", '_tmp.csv'))

# read smiles and create tmp file
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

with open(tmp_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for s in smiles_list:
        writer.writerow([s])

# generate ensemble config pointing to local checkpoint files (avoids HuggingFace downloads)
local_ensemble_config = {
    "electra": {
        "type": "electra",
        "ckpt_path": os.path.join(checkpoints_dir, "14ko0zcf_epoch=193.ckpt"),
        "target_labels_path": os.path.join(checkpoints_dir, "electra_classes.txt"),
        "classwise_weights_path": os.path.join(checkpoints_dir, "metrics_electra_14ko0zcf_80-10-10_short.json"),
    },
    "resgated": {
        "type": "resgated",
        "ckpt_path": os.path.join(checkpoints_dir, "0ps1g189_epoch=122.ckpt"),
        "target_labels_path": os.path.join(checkpoints_dir, "resgated_classes.txt"),
        "classwise_weights_path": os.path.join(checkpoints_dir, "metrics_0ps1g189_80-10-10_short.json"),
        "molecular_properties": [
            "chebai_graph.preprocessing.properties.AtomType",
            "chebai_graph.preprocessing.properties.NumAtomBonds",
            "chebai_graph.preprocessing.properties.AtomCharge",
            "chebai_graph.preprocessing.properties.AtomAromaticity",
            "chebai_graph.preprocessing.properties.AtomHybridization",
            "chebai_graph.preprocessing.properties.AtomNumHs",
            "chebai_graph.preprocessing.properties.BondType",
            "chebai_graph.preprocessing.properties.BondInRing",
            "chebai_graph.preprocessing.properties.BondAromaticity",
            "chebai_graph.preprocessing.properties.RDKit2DNormalized",
        ],
    },
    "chemlog_peptides": {"type": "chemlog_peptides"},
    "chemlog_element": {"type": "chemlog_element"},
    "chemlog_organox": {"type": "chemlog_organox"},
    "c3p": {
        "type": "c3p",
        "classwise_weights_path": os.path.join(checkpoints_dir, "c3p_trust.json"),
    },
    "chebi_lookup": {"type": "chebi_lookup"},
}
local_ensemble_config_path = "/tmp/ensemble_config_local.yml"
with open(local_ensemble_config_path, "w") as f:
    yaml.dump(local_ensemble_config, f)

# change working directory to a writable location before running the model
# (chemlog_extra writes relative path data/ from cwd at init time,
#  which fails if cwd is inside a read-only container image)
os.chdir("/tmp")

json_output_file = output_file.replace(".csv", ".json")

# run the model
predict(
    ensemble_config=local_ensemble_config_path,
    smiles=(),  # none inline
    smiles_file=tmp_file,
    output=json_output_file,
    ensemble_type="wmv-f1",
    use_confidence=True,
    resolve_inconsistencies=True
)

# read json output
output = json.load(open(json_output_file))
output_content = ['chebi_predicted_parents']
for smi in smiles_list:
    preds = output.get(smi)
    r = ";".join("CHEBI:" + o for o in sorted(preds)) if preds is not None else ""
    output_content.append(r)

# remove tmp files
os.remove(json_output_file)
os.remove(tmp_file)

# write output in a .csv file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    for row in output_content:
        writer.writerow([row])
