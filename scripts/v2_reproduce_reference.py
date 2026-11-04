"""Reproduce a frozen v1 headline run before interpreting v2 experiments."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prepare_data import load_labeled, load_curves_all
from src.splits import apply_split
from src.train import run_method_on_split, set_curve_context, set_run_meta
from src.provenance import freeze_git_provenance

OUT = ROOT / 'results' / 'v2' / 'reproduction'
OUT.mkdir(parents=True, exist_ok=True)
for name in ['features.py', 'train.py', 'models.py', 'metrics.py', 'constants.py']:
    shutil.copy2(ROOT/'src'/name, OUT/name)
freeze_git_provenance()
split=json.loads((ROOT/'data/splits/protocol1_seed42.json').read_text(encoding='utf-8'))
reference=json.loads((ROOT/'results/multirep/details/well_id/seed42_lgbm_global.json').read_text(encoding='utf-8'))
print('Loading exact v1 data and source; running full-row Global seed 42', flush=True)
df=load_labeled()
curves=load_curves_all()
set_curve_context(curves)
set_run_meta(split=split, config_path=ROOT/'configs/paper_r3_multirep.yaml')
result=run_method_on_split(apply_split(df,split), 'lgbm_global', seed=42)
(OUT/'seed42_lgbm_global.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
checks={key:{'expected':reference['overall'][key],'actual':result['overall'][key]}
        for key in ['macro_f1','accuracy','weighted_f1']}
for key in ['worst_well_f1','worst10pct_well_f1']:
    checks['fixed_'+key]={'expected':reference['well_summary_fixed'][key],'actual':result['well_summary_fixed'][key]}
passed=all(abs(v['expected']-v['actual'])<=1e-10 for v in checks.values())
report={'passed':passed,'tolerance':1e-10,'checks':checks,'source_hashes':{
    p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}}
(OUT/'gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
if not passed:
    raise SystemExit('Reproduction failed; do not interpret new runs.')
