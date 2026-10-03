"""FORCE 2020 lithofacies constants and shared paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
LAS_DIR = WORKSPACE / "Force_2020_all_wells_train_test_blind_hidden_final"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Protocol-1 primary multi-seed aggregation (closed-set). Seed 44 is open-set stress only.
# Historical 3-seed contrast (pre multi-rep revision); still used by release gate / SI tables.
PROTOCOL1_CLOSED_SEEDS: tuple[int, ...] = (42, 43, 45)
PROTOCOL1_OPENSET_SEEDS: tuple[int, ...] = (44,)
PROTOCOL1_ALL_SEEDS: tuple[int, ...] = (42, 43, 44, 45)

# Locked a priori closed-set rule (Protocol-1): a split enters primary aggregation iff
# the test partition has zero depths whose facies are absent from training
# (n_test_unseen_label_kept == 0); else open-set stress bucket (same spirit as
# seeds 42/43/45 vs 44). Bucket membership is computed from the split+labels,
# not from model scores.
#
# Predetermined 20-seed multi-rep schedule (includes historical 42--45).
# Do not reorder or cherry-pick after seeing scores.
PROTOCOL1_MULTIREP_SEEDS: tuple[int, ...] = (
    42,
    43,
    44,
    45,
    101,
    107,
    113,
    127,
    131,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
)

# Core LightGBM family for multi-rep primary analysis (not the full 16-method suite).
PROTOCOL1_CORE_METHODS: tuple[str, ...] = (
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_hybrid_norel",
    "lgbm_hybrid",
    "lgbm_hybrid_while",
    "lgbm_hybrid_smooth",
)

# W10 = mean of the worst ceil(0.1 * n_test_wells) wells under fixed taxonomy.
# With n_test_wells=17: ceil(1.7)=2 wells.

NULL_VALUE = -999.25

# Official FORCE 2020 lithology codes
LITHOLOGY_MAP = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

# Core wireline curves used across wells (others kept if present)
CORE_CURVES = [
    "GR",
    "RHOB",
    "NPHI",
    "DTC",
    "DTS",
    "PEF",
    "CALI",
    "RDEP",
    "RMED",
    "RSHA",
    "RXO",
    "SP",
    "ROP",
    "DRHO",
    "BS",
    "MUDWEIGHT",
]

META_COLS = [
    "well_id",
    "file_name",
    "block",  # NPD quadrant id (leading token of QUADRANT/BLOCK-WELL); not NPD block
    "depth",
    "depth_md",
    "x_loc",
    "y_loc",
    "z_loc",
    "lithology",
    "lithology_name",
    "confidence",
]

LABEL_COL = "lithology"
CONF_COL = "confidence"
WELL_COL = "well_id"
DEPTH_COL = "depth"
# Column name kept as ``block`` for parquet/CSV compatibility; value is NPD quadrant.
BLOCK_COL = "block"
QUADRANT_COL = BLOCK_COL
