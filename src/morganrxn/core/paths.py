from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT_DIR = project_root()
DATA_DIR = project_root() / "data"

RESULTS_DIR = ROOT_DIR / "results"
REACTION_RULES_DIR = DATA_DIR / "reaction_rules"

METANETX_DIR = DATA_DIR / "metanetx"
USPTO_DIR = DATA_DIR / "uspto"
