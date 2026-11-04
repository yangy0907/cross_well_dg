# Processed tables

Rebuild deterministically from FORCE~2020 LAS + labels:

```bash
py run_experiment.py prepare
```

This writes `curves_all.parquet` and `labeled.parquet`. SHA-256 digests are recorded in:

- `*.sha256` side-cars (written by `scripts/build_release_manifest.py`)
- every `results/summary_*.csv` column `hash_curves_all_parquet` / `hash_labeled_parquet`

Parquet binaries are gitignored by default (size); release deposits should ship the tables or the Zenodo archive together with these hashes.
