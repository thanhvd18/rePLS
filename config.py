
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any
import yaml

# ---- Load configuration from YAML ----

CONFIG_DIR: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file, with environment variable overrides."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Override with environment variables if set
    config['data_dir'] = os.getenv("REPLS_DATA_DIR", config.get('data_dir', 'data'))
    config['adni_processed_path'] = os.getenv("REPLS_ADNI_PATH", config.get('adni_processed_path', 'data/ADNI_FS_processed.csv'))
    config['oasis_processed_path'] = os.getenv("REPLS_OASIS_PATH", config.get('oasis_processed_path', 'data/OASIS_FS_processed.csv'))
    config['all_3_path'] = os.getenv("REPLS_ALL3_PATH", config.get('all_3_path', 'data/ADNI_FS_processed.csv'))

    return config


# Load configuration
_config = load_config()

# ---- Project paths (override-friendly) ----

BASE_DIR: Path = Path(__file__).resolve().parent

# Resolve data directory (support both absolute and relative paths)
if os.path.isabs(_config['data_dir']):
    DATA_DIR: Path = Path(_config['data_dir']).expanduser().resolve()
else:
    DATA_DIR: Path = (CONFIG_DIR / _config['data_dir']).expanduser().resolve()

# Resolve data file paths (support both absolute and relative paths)
def resolve_path(path_str: str) -> Path:
    """Resolve a path string to absolute Path, supporting both absolute and relative."""
    if os.path.isabs(path_str):
        return Path(path_str).expanduser().resolve()
    return (CONFIG_DIR / path_str).expanduser().resolve()


ADNI_PROCESSED_PATH: Path = resolve_path(_config['adni_processed_path'])
OASIS_PROCESSED_PATH: Path = resolve_path(_config['oasis_processed_path'])
ALL_3_PATH: Path = resolve_path(_config['all_3_path'])
SEVEN_NETWORK_PATH: Path = resolve_path(_config.get('seven_network_path', 'data/7network.csv'))

# ---- Domain config ----

networks: List[str] = _config.get('networks', ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"])
outcomes: List[str] = _config.get('outcomes', [
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "ADASQ4",
    "MMSE",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_perc_forgetting",
])