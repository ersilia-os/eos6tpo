"""
Hugging Face Api:
    - For Windows Users check: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#limitations

    Refer for Hugging Face Hub caching and versioning documentation:
        https://huggingface.co/docs/huggingface_hub/en/guides/download
        https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

root = os.path.dirname(os.path.abspath(__file__))


# def download_model_files_legacy(
#     model_config: dict[str, str | dict[str, str]],
# ) -> dict[str, Path]:
#     """
#     Downloads specified model files from a Hugging Face Hub repository using hf_hub_download.

#     Hugging Face Hub provides internal caching and versioning, so file management or duplication
#     checks are not required.

#     Args:
#         model_config (Dict[str, str | Dict[str, str]]): A dictionary containing:
#             - 'repo_id' (str): The Hugging Face repository ID (e.g., 'username/modelname').
#             - 'subfolder' (str): The subfolder within the repo where the files are located.
#             - 'files' (Dict[str, str]): A mapping from file type (e.g., 'ckpt_path', 'target_labels_path') to
#               actual file names (e.g., 'electra.ckpt', 'classes.txt').

#     Returns:
#         Dict[str, Path]: A dictionary mapping each file type to the local Path of the downloaded file.
#     """
#     repo_id = model_config["repo_id"]
#     subfolder = model_config.get("subfolder", None)
#     repo_type = model_config.get("repo_type", "model")
#     filenames = model_config["files"]

#     local_paths: dict[str, Path] = {}
#     for file_type, filename in filenames.items():
#         downloaded_file_path = hf_hub_download(
#             repo_id=repo_id,
#             filename=filename,
#             repo_type=repo_type,
#             subfolder=subfolder,
#         )
#         local_paths[file_type] = Path(downloaded_file_path)
#         print(f"\t Using file `{filename}` from: {downloaded_file_path}")

#     return local_paths

def download_model_files(
    model_config: dict[str, str | dict[str, str]],
) -> dict[str, Path]:
    """
    Always resolve file paths locally.
    Interprets relative paths as relative to the current working directory.
    """
    filenames = model_config["files"]

    local_paths: dict[str, Path] = {}
    for file_type, filename in filenames.items():
        p = os.path.join(root, "..", "..", "..", "..", "checkpoints", filename)
        p = Path(p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Local file not found: {p}")
        local_paths[file_type] = p
        print(f"\t Using LOCAL file `{filename}` -> {p}")

    return local_paths
