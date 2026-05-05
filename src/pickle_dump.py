"""Utility for dumping and loading the volleyball master JSON as a pickle file."""

import json
import pickle

from configs.path_config import JSON_DATA_DIR, PICKLE_DUMP_DIR


def dump_to_pickle() -> None:
    """
    Load volleyball_master.json and dump it as a pickle file.

    Acts as a singleton: skips dumping if the pickle file already exists
    under the PICKLE_DUMP_DIR path.
    """
    if PICKLE_DUMP_DIR.exists():
        print(f"Pickle file already exists at {PICKLE_DUMP_DIR}, skipping dump.")
        return

    # Ensure the parent directory exists
    PICKLE_DUMP_DIR.parent.mkdir(parents=True, exist_ok=True)

    with JSON_DATA_DIR.open("r") as json_file:
        data = json.load(json_file)

    with PICKLE_DUMP_DIR.open("wb") as pkl_file:
        pickle.dump(data, pkl_file)

    print(f"Successfully dumped pickle to {PICKLE_DUMP_DIR}")


def load_from_pickle() -> dict:
    """
    Load and return the dictionary from the dumped pickle file.

    Returns:
        dict: The volleyball master data dictionary.

    Raises:
        FileNotFoundError: If the pickle file does not exist.

    """
    if not PICKLE_DUMP_DIR.exists():
        raise FileNotFoundError(
            f"Pickle file not found at {PICKLE_DUMP_DIR}. "
            "Run dump_to_pickle() first.",
        )

    with PICKLE_DUMP_DIR.open("rb") as pkl_file:
        data = pickle.load(pkl_file)

    return data


if __name__ == "__main__":
    dump_to_pickle()
    data = load_from_pickle()
    print("Keys:", list(data.keys()))
