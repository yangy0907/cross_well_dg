"""Bounded, provenance-locked pilot: 16 core4 fits + 4 raw-mask full16 controls."""
from __future__ import annotations
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import yaml
from src.constants import CORE_CURVES
from src.provenance import freeze_git_provenance, hash_obj
from src.splits import apply_split
from src.train import run_method_on_split, set_curve_context, set_run_meta

CONFIG=ROOT/'configs/v2_pilot_common_core.yaml'
OUT=ROOT/'results/v2/pilot_common_core'
DATA=None
CURVES=None

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()

def init_worker():
    global DATA,CURVES
    cols=['well_id','depth','lithology','confidence']+CORE_CURVES
    DATA=pd.read_parquet(ROOT/'data/processed/labeled.parquet',columns=cols)
    CURVES=pd.read_parquet(ROOT/'data/processed/curves_all.parquet',columns=['well_id','depth']+CORE_CURVES)
    set_curve_context(CURVES)
    freeze_git_provenance()

def run_case(case, fingerprint):
    seed,curve_set,method=case
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    split=json.loads((ROOT/f'data/splits/protocol1_seed{seed}.json').read_text(encoding='utf-8'))
    set_run_meta(split=split,config_path=CONFIG)
    result=run_method_on_split(apply_split(DATA,split),method,seed=seed,
        curves=cfg[curve_set],raw_missingness_mask=True,model_n_jobs=cfg['n_jobs'])
    reference=json.loads((ROOT/f'results/multirep/details/well_id/seed{seed}_lgbm_global.json').read_text(encoding='utf-8'))
    for key in ['n_train','n_val','n_test','n_test_unseen_label_kept','n_fixed_labels']:
        if result[key]!=reference[key]:raise AssertionError(f'{case}: unmatched {key}')
    if result['n_train_fit']!=reference['n_train']:raise AssertionError('Row budget was changed')
    result.update({'study':'v2_pilot_common_core_20261103','curve_set':curve_set,
        'execution_fingerprint':fingerprint,'evidence_origin':'new_fit',
        'matching_reference':str(ROOT/f'results/multirep/details/well_id/seed{seed}_lgbm_global.json')})
    p=OUT/'details'/f'{curve_set}_seed{seed}_{method}.json'
    temp=p.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(p)
    return {'case':case,'seconds':result['seconds'],'macro':result['overall']['macro_f1'],
            'fixed_worst':result['well_summary_fixed']['worst_well_f1']}

def main():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    gate=json.loads((ROOT/'results/v2/reproduction/gate.json').read_text(encoding='utf-8'))
    if not gate['passed']:raise RuntimeError('Reproduction gate has not passed')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'details').mkdir(exist_ok=True)
    files=[ROOT/'src'/n for n in ['train.py','features.py','models.py','metrics.py','constants.py','provenance.py']]+[Path(__file__),CONFIG,ROOT/'requirements.lock.txt']
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    source_hashes.update({f'data/splits/protocol1_seed{s}.json':digest(ROOT/f'data/splits/protocol1_seed{s}.json') for s in cfg['seeds']})
    source_hashes.update({f'data/processed/{n}':digest(ROOT/'data/processed'/n) for n in ['labeled.parquet','curves_all.parquet']})
    fingerprint=hash_obj(source_hashes)
    meta={'execution_fingerprint':fingerprint,'source_hashes':source_hashes,'config':cfg,
          'started_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'python':sys.version,'n_workers':2,
          'interpretation':'Exploratory split sensitivity, not population inference; full16 non-WellNorm references are not new fits.'}
    meta_path=OUT/'execution_manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8'))['execution_fingerprint']!=fingerprint:
        raise RuntimeError('Source/config/data changed. Use a different output directory rather than mixing runs.')
    else:
        meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    for p in files:
        q=OUT/'execution_source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    cases=[(s,c,m) for s in cfg['seeds'] for c in ['core4','full16'] for m in cfg['new_fit_schedule'][c]]
    pending=[]
    for case in cases:
        s,c,m=case;p=OUT/'details'/f'{c}_seed{s}_{m}.json'
        if p.exists():
            if json.loads(p.read_text(encoding='utf-8')).get('execution_fingerprint')!=fingerprint:
                raise RuntimeError(f'Incompatible completed row: {p}')
        else:pending.append(case)
    print(f'Pilot locked: {len(cases)} total fits; {len(pending)} pending; two workers, {cfg["n_jobs"]} threads each.',flush=True)
    lock=OUT/'.running.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        with futures.ProcessPoolExecutor(max_workers=2,initializer=init_worker) as pool:
            jobs={pool.submit(run_case,c,fingerprint):c for c in pending}
            for f in futures.as_completed(jobs):
                record=f.result()
                print(json.dumps(record,ensure_ascii=False),flush=True)
                with (OUT/'progress.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps(record,ensure_ascii=False)+'\n')
        (OUT/'COMPLETE.json').write_text(json.dumps({'complete':True,'new_fit_count':len(cases),'execution_fingerprint':fingerprint}),encoding='utf-8')
        print('COMPLETE: all 20 fits saved independently.',flush=True)
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
