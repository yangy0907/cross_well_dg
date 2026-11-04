"""Resume inference from all twelve saved models; preserve float64 physical depths.

FeatureBundle stores export depths as float32, whereas relative depth is built
from the original float64 dataframe. No model is fitted by this recovery script.
"""
from pathlib import Path
import sys
import json
import hashlib
import concurrent.futures as futures
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.features import FeaturePipeline
from src.splits import apply_split
from scripts import v3_depth_validation as study

OUT = study.OUT


def run(case):
    seed, curves = case
    manifest = json.loads((OUT / 'execution_manifest.json').read_text(encoding='utf-8'))
    cfg = manifest['config']
    split = json.loads((OUT / 'splits' / f'parent_seed{seed}.json').read_text(encoding='utf-8'))
    parts = apply_split(study.DATA, split)
    pipe = FeaturePipeline(mode='hybrid', curves=cfg['curve_sets'][curves],
        use_relative_depth=True, raw_missingness_mask=True, curve_context=study.CONTEXT)
    tr = pipe.fit_transform(parts['train']); del tr
    te = pipe.transform(parts['test'])
    depths = parts['test'].depth.to_numpy(dtype=np.float64)
    base = OUT / 'details' / f'{curves}_seed{seed}'
    h = json.loads(Path(str(base) + '_hybrid.json').read_text(encoding='utf-8'))
    mapping = {int(k): v for k, v in h['label_mapping'].items()}
    hp = pd.read_parquet(Path(str(base) + '_hybrid.predictions.parquet'))
    npred = pd.read_parquet(Path(str(base) + '_norel.predictions.parquet'))
    for p in [hp, npred]:
        np.testing.assert_array_equal(te.wells, p.well_id.to_numpy())
        np.testing.assert_array_equal(te.depths, p.depth.to_numpy())
    y = hp.y_true_encoded.to_numpy(dtype=np.int32)
    pred = hp.y_pred_encoded.to_numpy(dtype=np.int32)
    n = npred.y_pred_encoded.to_numpy(dtype=np.int32)
    model = lgb.Booster(model_str=Path(str(base) + '_hybrid.model.txt').read_text(encoding='utf-8'))
    restored = np.argmax(model.predict(te.X, num_threads=8), axis=1).astype(np.int32)
    np.testing.assert_array_equal(restored, pred)
    bounds = study.CONTEXT.groupby('well_id').depth.agg(['min', 'max'])
    lo = pd.Series(te.wells).map(bounds['min']).to_numpy()
    hi = pd.Series(te.wells).map(bounds['max']).to_numpy()
    frame = pd.DataFrame({'well_id': te.wells, 'depth': depths,
        'y_true_encoded': y, 'baseline_hybrid': pred, 'hybrid_norel': n})
    records = []
    for crop in cfg['boundary_crops']:
        keep, rel = study.boundary_coordinates(depths, lo, hi, crop['top'], crop['bottom'])
        if crop['name'] == 'baseline':
            np.testing.assert_array_equal(rel, te.X[:, te.feature_names.index('rel_depth')])
            changed_pred = pred[keep]
        else:
            changed = study.change_relative_column(te.X[keep], te.feature_names, rel[keep])
            changed_pred = np.argmax(model.predict(changed, num_threads=8), axis=1).astype(np.int32)
            exported = np.full(len(y), -1, dtype=np.int32)
            exported[keep] = changed_pred
            frame[crop['name']] = exported
        records.append({'scenario': crop['name'], 'top_crop': crop['top'], 'bottom_crop': crop['bottom'],
            'n_scored': int(keep.sum()), 'n_scored_wells': len(np.unique(te.wells[keep])),
            'n_unseen_retained': int((y[keep] == len(mapping)).sum()),
            'hybrid_changed': study.serial_evaluation(y[keep], changed_pred, te.wells[keep], mapping),
            'hybrid_same_rows': study.serial_evaluation(y[keep], pred[keep], te.wells[keep], mapping),
            'norel_same_rows': study.serial_evaluation(y[keep], n[keep], te.wells[keep], mapping)})
    frame.to_parquet(Path(str(base) + '_boundary_predictions.parquet'), index=False)
    study.atomic_json(Path(str(base) + '_stress.json'), {'seed': seed, 'curve_set': curves,
        'execution_fingerprint': manifest['execution_fingerprint'], 'records': records,
        'scope': cfg['interpretation'], 'original_depth_precision': 'float64',
        'saved_model_full_baseline_predictions_identical': True,
        'recovery_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return {'seed': seed, 'curve_set': curves, 'boundary_scenarios': len(records), 'new_fits': 0}


def main():
    manifest = json.loads((OUT / 'execution_manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['source_hashes'].items():
        if study.digest(ROOT / rel) != expected:
            raise RuntimeError(f'Frozen training input changed: {rel}')
    cases = [(s, c) for s in manifest['config']['seeds'] for c in ['core4', 'full16']]
    with futures.ProcessPoolExecutor(max_workers=2, initializer=study.init_worker) as pool:
        for result in pool.map(run, cases):
            print(json.dumps(result), flush=True)
    study.atomic_json(OUT / 'inference_recovery.json', {
        'cause': 'Float32 exported FeatureBundle depths do not exactly reconstruct float64-engineered relative depth.',
        'resolution': 'Reconstruct features and boundary coordinates from original float64 dataframe depths.',
        'new_fit_count': 0, 'all_twelve_models_previously_saved': True,
        'full_baseline_predictions_identical': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    study.atomic_json(OUT / 'COMPLETE.json', {'complete': True, 'new_split_fit_count': 12,
        'gate_fit_count': 1, 'prior_failed_gate_fit_count': 1, 'phase_total_fit_attempts': 14,
        'boundary_additional_fits': 0, 'execution_fingerprint': manifest['execution_fingerprint']})


if __name__ == '__main__':
    main()
