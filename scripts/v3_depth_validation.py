"""Bounded depth validation with persisted models, paired boundary controls and provenance.

The narrow fit path reuses the original feature/model/metric classes. A one-fit
gate must reproduce the completed v2 core4 Hybrid reference before new splits run.
No v1/v2 source, split or numerical result is overwritten.
"""
from __future__ import annotations
import concurrent.futures as futures
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from src.constants import CORE_CURVES, LITHOLOGY_MAP
from src.features import FeaturePipeline
from src.metrics import evaluate_split
from src.models import LGBMConfig, LGBMLithologyModel
from src.provenance import hash_obj
from src.splits import apply_split, protocol1_parent_grouped_wells

CONFIG = ROOT / 'configs/v3_relative_depth_validation.yaml'
OUT = ROOT / 'results/v3/relative_depth_validation_run2'
DATA = None
CONTEXT = None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def init_worker():
    global DATA, CONTEXT
    DATA = pd.read_parquet(ROOT / 'data/processed/labeled.parquet',
                           columns=['well_id', 'depth', 'lithology', 'confidence'] + CORE_CURVES)
    CONTEXT = pd.read_parquet(ROOT / 'data/processed/curves_all.parquet',
                              columns=['well_id', 'depth'] + CORE_CURVES)


def serial_evaluation(y, pred, wells, mapping):
    labels = {v: LITHOLOGY_MAP.get(k, str(k)) for k, v in mapping.items()}
    if (y == len(mapping)).any():
        labels[len(mapping)] = 'UNSEEN_IN_TRAIN'
    result = evaluate_split(y, pred, wells, label_names=labels,
                            pooled_labels=sorted(np.unique(y).tolist()),
                            fixed_well_labels=sorted(mapping.values()))
    for key in ['well_table', 'well_table_fixed']:
        result[key] = result[key].to_dict(orient='records')
    result['confusion_matrix'] = result['confusion_matrix'].tolist()
    return result


def fit_case(split, curve_set, method, cfg):
    t0 = time.monotonic()
    parts = apply_split(DATA, split)
    pipe = FeaturePipeline(mode='hybrid', curves=cfg['curve_sets'][curve_set],
                           use_relative_depth=(method == 'lgbm_hybrid'),
                           raw_missingness_mask=True, curve_context=CONTEXT)
    tr = pipe.fit_transform(parts['train'])
    va = pipe.transform(parts['val'])
    te = pipe.transform(parts['test'])
    classes = sorted(int(v) for v in np.unique(tr.y) if v >= 0)
    mapping = {v: i for i, v in enumerate(classes)}
    val_mask = np.array([int(v) in mapping for v in va.y])
    ytr = np.array([mapping[int(v)] for v in tr.y], dtype=np.int32)
    yva = np.array([mapping[int(v)] for v in va.y[val_mask]], dtype=np.int32)
    yte = np.array([mapping.get(int(v), len(mapping)) for v in te.y], dtype=np.int32)
    model_cfg = LGBMConfig(random_state=split['seed'], n_estimators=cfg['n_estimators'],
                           n_jobs=cfg['n_jobs'])
    model = LGBMLithologyModel(config=model_cfg)
    model.fit(tr.X, ytr, tr.conf, tr.wells, X_val=va.X[val_mask], y_val=yva)
    pred = model.predict(te.X).astype(np.int32)
    ev = serial_evaluation(yte, pred, te.wells, mapping)
    result = {'seed': split['seed'], 'curve_set': curve_set, 'method': method,
              'pipeline_version': 'v3_relative_depth_validation_20261103',
              'n_train': len(tr.y), 'n_train_fit': len(tr.y), 'n_val': int(val_mask.sum()),
              'n_val_total': len(va.y), 'n_test': len(te.y),
              'n_test_unseen_label_kept': int((yte == len(mapping)).sum()),
              'label_regime': 'open_set' if (yte == len(mapping)).any() else 'closed_set',
              'n_features': len(tr.feature_names), 'feature_names': tr.feature_names,
              'label_mapping': {str(k): v for k, v in mapping.items()},
              'train_class_names': [LITHOLOGY_MAP.get(k, str(k)) for k in classes],
              'learner_config': asdict(model_cfg), 'seconds': round(time.monotonic() - t0, 2), **ev}
    return result, model, pipe, te, yte, pred, mapping


def boundary_coordinates(depths, dmin, dmax, top, bottom):
    """Coordinates and common support; rows outside observable interval are excluded."""
    if top < 0 or bottom < 0 or top + bottom >= 1:
        raise ValueError('Invalid crop')
    depths, dmin, dmax = [np.asarray(x, dtype=np.float64) for x in [depths, dmin, dmax]]
    span = dmax - dmin
    lo = dmin + top * span
    hi = dmax - bottom * span
    keep = (depths >= lo) & (depths <= hi)
    rel = ((depths - lo) / (hi - lo + 1e-6)).astype(np.float32)
    return keep, rel


def change_relative_column(X, feature_names, rel):
    out = X.copy()
    out[:, feature_names.index('rel_depth')] = rel
    return out


def save_fit(base, fit, fingerprint):
    result, model, pipe, te, y, pred, mapping = fit
    result['execution_fingerprint'] = fingerprint
    # LightGBM's native Windows file API cannot reliably open Unicode paths.
    model_path = base.with_suffix('.model.txt')
    model_path.write_text(model.model.booster_.model_to_string(), encoding='utf-8')
    restored = lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
    n_check = min(2048, len(y))
    np.testing.assert_allclose(restored.predict(te.X[:n_check], num_threads=model.config.n_jobs),
                               model.predict_proba(te.X[:n_check]), atol=1e-12, rtol=1e-12)
    result['saved_model_probability_readback_rows'] = n_check
    # Store fitted transforms without serializing the million-row context table.
    transform_state = {'feature_names': te.feature_names,
                       'global_median': pipe._global_median.to_dict(),
                       'global_std': pipe._global_std.to_dict(),
                       'imputer_statistics': pipe._imputer.statistics_.tolist(),
                       'context_source': 'data/processed/curves_all.parquet'}
    atomic_json(base.with_suffix('.transforms.json'), transform_state)
    pd.DataFrame({'well_id': te.wells, 'depth': te.depths, 'y_true_encoded': y,
                  'y_pred_encoded': pred}).to_parquet(base.with_suffix('.predictions.parquet'), index=False)
    atomic_json(base.with_suffix('.json'), result)


def run_pair(case, fingerprint):
    seed, curve_set = case
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    split = json.loads((OUT / 'splits' / f'parent_seed{seed}.json').read_text(encoding='utf-8'))
    base = OUT / 'details' / f'{curve_set}_seed{seed}'
    hpath = Path(str(base) + '_hybrid.json')
    npath = Path(str(base) + '_norel.json')
    stresspath = Path(str(base) + '_stress.json')
    # Each pair is a unit: never combine a baseline from another execution.
    if all(p.exists() for p in [hpath, npath, stresspath]):
        for p in [hpath, npath, stresspath]:
            if json.loads(p.read_text(encoding='utf-8'))['execution_fingerprint'] != fingerprint:
                raise RuntimeError('Incompatible saved pair')
        return {'seed': seed, 'curve_set': curve_set, 'reused_completed_pair': True}
    nfit = fit_case(split, curve_set, 'lgbm_hybrid_norel', cfg)
    save_fit(Path(str(base) + '_norel'), nfit, fingerprint)
    # Release source/validation feature matrices automatically; keep test identity/predictions.
    nresult, _, _, nte, ny, npred, nmapping = nfit
    del nfit
    hfit = fit_case(split, curve_set, 'lgbm_hybrid', cfg)
    hresult, model, pipe, te, y, pred, mapping = hfit
    for key in ['n_train', 'n_train_fit', 'n_val', 'n_test', 'n_test_unseen_label_kept']:
        if nresult[key] != hresult[key]:
            raise AssertionError(f'Mismatched paired budget: {key}')
    if mapping != nmapping:
        raise AssertionError('Class mapping mismatch')
    for a, b in [(te.wells, nte.wells), (te.depths, nte.depths), (y, ny)]:
        np.testing.assert_array_equal(a, b)
    hcols = [i for i, name in enumerate(te.feature_names) if name != 'rel_depth']
    if [te.feature_names[i] for i in hcols] != nte.feature_names:
        raise AssertionError('Non-depth feature order changed')
    np.testing.assert_array_equal(te.X[:, hcols], nte.X)
    save_fit(Path(str(base) + '_hybrid'), hfit, fingerprint)
    bounds = CONTEXT.groupby('well_id').depth.agg(['min', 'max'])
    dmin = pd.Series(te.wells).map(bounds['min']).to_numpy()
    dmax = pd.Series(te.wells).map(bounds['max']).to_numpy()
    rel_i = te.feature_names.index('rel_depth')
    records = []
    prediction_frame = pd.DataFrame({'well_id': te.wells, 'depth': te.depths,
                                     'y_true_encoded': y, 'baseline_hybrid': pred,
                                     'hybrid_norel': npred})
    for crop in cfg['boundary_crops']:
        keep, rel = boundary_coordinates(te.depths, dmin, dmax, crop['top'], crop['bottom'])
        if crop['name'] == 'baseline':
            np.testing.assert_array_equal(rel, te.X[:, rel_i])
            stressed = pred
        else:
            # Only one feature changes. Source fit, global/well-z/masks stay frozen.
            changed = change_relative_column(te.X[keep], te.feature_names, rel[keep])
            stressed = model.predict(changed).astype(np.int32)
            all_pred = np.full(len(y), -1, dtype=np.int32)
            all_pred[keep] = stressed
            prediction_frame[crop['name']] = all_pred
        current = pred if crop['name'] == 'baseline' else stressed
        ev_stress = serial_evaluation(y[keep], current[keep] if crop['name'] == 'baseline' else current,
                                      te.wells[keep], mapping)
        ev_base = serial_evaluation(y[keep], pred[keep], te.wells[keep], mapping)
        ev_norel = serial_evaluation(y[keep], npred[keep], te.wells[keep], mapping)
        records.append({'scenario': crop['name'], 'top_crop': crop['top'], 'bottom_crop': crop['bottom'],
                        'n_scored': int(keep.sum()), 'n_scored_wells': len(np.unique(te.wells[keep])),
                        'n_unseen_retained': int((y[keep] == len(mapping)).sum()),
                        'hybrid_changed': ev_stress, 'hybrid_same_rows': ev_base,
                        'norel_same_rows': ev_norel})
    prediction_frame.to_parquet(Path(str(base) + '_boundary_predictions.parquet'), index=False)
    atomic_json(stresspath, {'execution_fingerprint': fingerprint, 'seed': seed,
                             'curve_set': curve_set, 'records': records,
                             'scope': cfg['interpretation']})
    return {'seed': seed, 'curve_set': curve_set,
            'regime': hresult['label_regime'], 'norel_macro': nresult['overall']['macro_f1'],
            'hybrid_macro': hresult['overall']['macro_f1'],
            'relative_depth_delta': hresult['overall']['macro_f1'] - nresult['overall']['macro_f1']}


def reproduction_gate(cfg, fingerprint):
    path = OUT / 'gate.json'
    if path.exists():
        gate = json.loads(path.read_text(encoding='utf-8'))
        if gate['execution_fingerprint'] != fingerprint or not gate['passed']:
            raise RuntimeError('Incompatible gate')
        return
    init_worker()
    split = json.loads((ROOT / 'data/splits/protocol1_seed42.json').read_text(encoding='utf-8'))
    fit = fit_case(split, 'core4', 'lgbm_hybrid', cfg)
    result = fit[0]
    old = json.loads((ROOT / 'results/v2/pilot_common_core/details/core4_seed42_lgbm_hybrid.json').read_text(encoding='utf-8'))
    for key in ['n_train', 'n_train_fit', 'n_val', 'n_test', 'n_test_unseen_label_kept', 'n_features']:
        if result[key] != old[key]:
            raise AssertionError(f'Reproduction budget differs: {key}')
    checks = {}
    for section, keys in [('overall', ['macro_f1', 'accuracy', 'weighted_f1']),
                          ('well_summary_fixed', ['worst_well_f1', 'worst10pct_well_f1'])]:
        for key in keys:
            checks[key] = {'reference': old[section][key], 'new': result[section][key],
                           'absolute_error': abs(old[section][key] - result[section][key])}
    gate = {'passed': all(c['absolute_error'] <= 1e-10 for c in checks.values()),
            'execution_fingerprint': fingerprint, 'checks': checks, 'fit_executions': 1}
    save_fit(OUT / 'gate_seed42_core4_hybrid', fit, fingerprint)
    atomic_json(path, gate)
    if not gate['passed']:
        raise AssertionError(gate)
    print('GATE PASSED: five v2 metrics reproduced to 1e-10.', flush=True)


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    for name in ['splits', 'details', 'execution_source']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    lock = OUT / '.running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        meta_data = pd.read_parquet(ROOT / 'data/processed/labeled.parquet', columns=['well_id', 'lithology'])
        regimes = []
        for seed in cfg['seeds']:
            split = protocol1_parent_grouped_wells(meta_data, seed=seed, save=False,
                       train_frac=cfg['train_frac'], val_frac=cfg['val_frac'])
            p = OUT / 'splits' / f'parent_seed{seed}.json'
            if p.exists() and json.loads(p.read_text(encoding='utf-8')) != split:
                raise RuntimeError('Saved split differs')
            atomic_json(p, split)
            train_labels = set(meta_data.loc[meta_data.well_id.isin(split['train_wells']), 'lithology'])
            test = meta_data.loc[meta_data.well_id.isin(split['test_wells']), 'lithology']
            unseen = int((~test.isin(train_labels)).sum())
            regimes.append({'seed': seed, 'unseen_test_rows': unseen,
                            'regime': 'open_set' if unseen else 'closed_set'})
        files = [Path(__file__), CONFIG, ROOT / 'requirements.lock.txt']
        files += [ROOT / 'tests/test_v3_depth_validation.py', ROOT / 'results/v3/ANALYSIS_PLAN.md']
        files += [ROOT / 'src' / n for n in ['features.py', 'models.py', 'metrics.py', 'splits.py', 'constants.py', 'provenance.py']]
        files += list((OUT / 'splits').glob('*.json'))
        files += [ROOT / 'data/processed' / n for n in ['labeled.parquet', 'curves_all.parquet']]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
        hashes['results/v2/pilot_common_core/details/core4_seed42_lgbm_hybrid.json'] = digest(ROOT / 'results/v2/pilot_common_core/details/core4_seed42_lgbm_hybrid.json')
        fingerprint = hash_obj(hashes)
        manifest_path = OUT / 'execution_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['execution_fingerprint'] != fingerprint:
                raise RuntimeError('Frozen execution inputs changed')
        else:
            atomic_json(manifest_path, {'execution_fingerprint': fingerprint, 'source_hashes': hashes,
                        'config': cfg, 'regimes': regimes, 'python': sys.version,
                        'locked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                        'planned_new_split_fits': 12, 'planned_gate_fits': 1,
                        'prior_failed_gate_fit_count': cfg['prior_failed_gate_fit_count']})
        for p in files:
            if p.suffix == '.parquet':
                continue
            q = OUT / 'execution_source' / p.relative_to(ROOT)
            q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
        print('LOCKED: ' + json.dumps(regimes), flush=True)
        reproduction_gate(cfg, fingerprint)
        # Gate data are not needed in the parent after worker data are loaded.
        global DATA, CONTEXT
        DATA = None; CONTEXT = None
        cases = [(s, c) for s in cfg['seeds'] for c in cfg['curve_sets']]
        with futures.ProcessPoolExecutor(max_workers=cfg['n_workers'], initializer=init_worker) as pool:
            jobs = [pool.submit(run_pair, c, fingerprint) for c in cases]
            for job in futures.as_completed(jobs):
                record = job.result()
                print(json.dumps(record), flush=True)
                with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record) + '\n')
        # Source/config/data remain locked throughout fitting and inference.
        for rel, expected in hashes.items():
            if digest(ROOT / rel) != expected:
                raise RuntimeError(f'Frozen input changed: {rel}')
        atomic_json(OUT / 'COMPLETE.json', {'complete': True, 'new_split_fit_count': 12,
                    'gate_fit_count': 1, 'boundary_additional_fits': 0,
                    'prior_failed_gate_fit_count': cfg['prior_failed_gate_fit_count'],
                    'phase_total_fit_attempts': 12 + 1 + cfg['prior_failed_gate_fit_count'],
                    'execution_fingerprint': fingerprint})
        print('COMPLETE: 12 new-split fits + 1 gate; boundary checks reuse models.', flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
