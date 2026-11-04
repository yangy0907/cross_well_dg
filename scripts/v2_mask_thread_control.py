"""One diagnostic fit at the same thread count, after the bounded pilot completes."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.splits import apply_split
from src.train import run_method_on_split,set_curve_context,set_run_meta
from src.provenance import freeze_git_provenance

def main():
    pilot=ROOT/'results/v2/pilot_common_core'
    if not (pilot/'COMPLETE.json').exists():raise RuntimeError('Finish the pilot before the extra diagnostic fit')
    out=ROOT/'results/v2/mask_thread_control';out.mkdir(parents=True,exist_ok=True)
    cfgpath=ROOT/'configs/v2_mask_thread_control.yaml'
    cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8'))
    output=out/'seed42_legacy_mask_8threads.json'
    if output.exists():
        result=json.loads(output.read_text(encoding='utf-8'))
    else:
        cols=['well_id','depth','lithology','confidence']+cfg['curves']
        df=pd.read_parquet(ROOT/'data/processed/labeled.parquet',columns=cols)
        curves=pd.read_parquet(ROOT/'data/processed/curves_all.parquet',columns=['well_id','depth']+cfg['curves'])
        split=json.loads((ROOT/'data/splits/protocol1_seed42.json').read_text(encoding='utf-8'))
        freeze_git_provenance();set_curve_context(curves);set_run_meta(split=split,config_path=cfgpath)
        result=run_method_on_split(apply_split(df,split),cfg['method'],seed=42,
            curves=cfg['curves'],raw_missingness_mask=False,model_n_jobs=8)
        result['control_purpose']=cfg['purpose']
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    old=json.loads((ROOT/'results/multirep/details/well_id/seed42_lgbm_wellnorm.json').read_text(encoding='utf-8'))
    corrected=json.loads((pilot/'details/full16_seed42_lgbm_wellnorm.json').read_text(encoding='utf-8'))
    metrics={}
    for name,section,key in [('macro','overall','macro_f1'),('worst_fixed','well_summary_fixed','worst_well_f1')]:
        metrics[name]={'v1_legacy':old[section][key],'legacy_8threads':result[section][key],
                       'raw_mask_8threads':corrected[section][key],
                       'mask_delta_same_threads':corrected[section][key]-result[section][key]}
    report={'legacy_thread_equivalence':all(abs(v['v1_legacy']-v['legacy_8threads'])<=1e-10 for v in metrics.values()),
            'metrics':metrics,'scope':'One diagnostic seed. Not a multi-seed significance test.',
            'config_hash':hashlib.sha256(cfgpath.read_bytes()).hexdigest()}
    (out/'control_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
