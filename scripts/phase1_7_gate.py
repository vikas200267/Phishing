from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'command':' '.join(cmd),'returncode':r.returncode,'stdout':r.stdout[-4000:],'stderr':r.stderr[-4000:]}

def main():
    checks=[]
    checks.append(run([sys.executable,'-m','compileall','-q','app','scripts']))
    checks.append(run([sys.executable,'-m','pytest','-q']))
    checks.append(run([sys.executable,'scripts/production_doctor.py']))
    temporal=run([sys.executable,'scripts/temporal_gate.py','data/training/production_training.csv'])
    temporal['required_for_phase_gate']=False
    checks.append(temporal)
    checks.append(run([sys.executable,'scripts/audit_dataset_leakage.py','--train','data/training/production_training.csv','--test','data/realworld/archive/openphish_20261005_unseen.csv']))
    required=[x for x in checks if x.get('required_for_phase_gate', True)]
    payload={'phase':'1-7','checks':checks,'passed':all(x['returncode']==0 for x in required),'temporal_status': next((x for x in checks if 'temporal_gate.py' in x.get('command','')), None)}
    out=ROOT/'data/manifests/phase1_7_gate.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2));raise SystemExit(0 if payload['passed'] else 1)
if __name__=='__main__':main()
