import json
from pathlib import Path

# The main project folder (two levels above this file)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Which knowledge file belongs to which dataset
KB_FILES = {
    "ciciot2023": REPO_ROOT / "data" / "kb" / "ciciot2023_kb.json",
    "edgeiiotset": REPO_ROOT / "data" / "kb" / "edgeiiotset_kb.json",
}


def get_description(predicted_class, dataset="ciciot2023"):
    """Return the description for a predicted attack class."""
    with open(KB_FILES[dataset], encoding="utf-8") as f:
        kb = json.load(f)
    if predicted_class not in kb:
        raise KeyError(f"'{predicted_class}' not found in the {dataset} knowledge file")
    return kb[predicted_class]


if __name__ == "__main__":
    # Quick test: run this file directly to see it work
    print(get_description("BenignTraffic", "ciciot2023"))
    print()
    print(get_description("DoS-SYN_Flood", "ciciot2023"))
    print()
    print(get_description("Normal", "edgeiiotset"))