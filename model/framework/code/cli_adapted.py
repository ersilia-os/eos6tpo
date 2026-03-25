from chebifier.model_registry import ENSEMBLES
import importlib.resources
import yaml


def predict(
    ensemble_config,
    smiles,
    smiles_file,
    output,
    ensemble_type,
    use_confidence,
    resolve_inconsistencies=True,
):
    """Predict ChEBI classes for SMILES strings using an ensemble model."""
    # Load configuration from YAML file
    if not ensemble_config:
        print("Using default ensemble configuration")
        with (
            importlib.resources.files("chebifier")
            .joinpath("ensemble.yml")
            .open("r") as f
        ):
            config = yaml.safe_load(f)
    else:
        print(f"Loading ensemble configuration from {ensemble_config}")
        with open(ensemble_config, "r") as f:
            config = yaml.safe_load(f)

    with (
        importlib.resources.files("chebifier")
        .joinpath("model_registry.yml")
        .open("r") as f
    ):
        model_registry = yaml.safe_load(f)

    new_config = {}
    for model_name, entry in config.items():
        if "load_model" in entry:
            if entry["load_model"] not in model_registry:
                raise ValueError(
                    f"Model {entry['load_model']} not found in model registry. "
                    f"Available models are: {','.join(model_registry.keys())}."
                )
            new_config[model_name] = {**model_registry[entry["load_model"]], **entry}
        else:
            new_config[model_name] = entry
    config = new_config

    # Instantiate ensemble model
    ensemble = ENSEMBLES[ensemble_type](
        config,
        resolve_inconsistencies=resolve_inconsistencies,
    )

    # Collect SMILES strings from arguments and/or file
    smiles_list = list(smiles)
    if smiles_file:
        with open(smiles_file, "r") as f:
            smiles_list.extend([line.strip() for line in f if line.strip() and line.strip() != 'smiles'])

    if not smiles_list:
        print("No SMILES strings provided. Use --smiles or --smiles-file options.")
        return

    # Make predictions in batches to avoid OOM with large inputs
    BATCH_SIZE = 100
    all_predictions = {}
    for i in range(0, len(smiles_list), BATCH_SIZE):
        batch = smiles_list[i:i + BATCH_SIZE]
        batch_preds = ensemble.predict_smiles_list(batch, use_confidence=use_confidence)
        for smi, pred in zip(batch, batch_preds):
            all_predictions[smi] = pred

    if output:
        import json

        with open(output, "w") as f:
            json.dump(all_predictions, f, indent=2)

    else:
        for smiles, prediction in all_predictions.items():
            print(f"Result for: {smiles}")
            if prediction:
                print(f"  Predicted classes: {', '.join(map(str, prediction))}")
            else:
                print("  No predictions")
