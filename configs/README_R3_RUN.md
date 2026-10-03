# Canonical r3 release schedule — run ONLY this (resume-safe; skips finished methods).
# Do NOT pass --protocol / --methods unless you also pass --allow-partial.
#
# Usage (from cross_well_dg/):
#   py -u run_experiment.py run --config configs/paper_r3_release.yaml
#
# paper_r3_release.yaml already includes WellNorm-norel and equal-row RF-full / XGB.
# Do NOT chain paper_equal_budget_p1.yaml or paper_r3_wellnorm_norel.yaml into the
# same summary_*.csv (resume would wipe foreign config_hash rows).
#
# After this finishes:
#   py -u scripts/matched_leakage_study.py
#   py -u scripts/revision_r3_openset_bootstrap.py
#   py -u scripts/embargo_fixed_block_study.py
#   py -u scripts/while_coldstart_study.py
#   py -u scripts/make_paper_figures.py
#   py -u scripts/check_active_results_r3.py
