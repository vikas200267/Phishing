from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse, hashlib, json
import os, tempfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import joblib
from app.registry import promote, current
from app.url_model import load_bundle
from app.train import evaluate, url_series
from scripts.audit_dataset_leakage import load as load_url_domain_sets, registrable

ROOT=Path(__file__).resolve().parents[1]



def _load_candidate(path: Path):
    complete=path/'COMPLETE'; meta_path=path/'metadata.json'; feature_path=path/'features.json'; model_path=path/'model.joblib'
    if not all(p.exists() for p in (complete,meta_path,feature_path,model_path)):
        raise RuntimeError(f'incomplete candidate model: {path}')
    meta=json.loads(meta_path.read_text())
    if meta.get('version') != path.name or meta.get('schema_version') != 6:
        raise RuntimeError(f'invalid candidate metadata: {path}')
    digest_path=path/'model.sha256'
    if digest_path.exists():
        import hashlib
        expected=digest_path.read_text().strip().split()[0]
        actual=hashlib.sha256(model_path.read_bytes()).hexdigest()
        if expected != actual: raise RuntimeError(f'candidate integrity mismatch: {path.name}')
    bundle=joblib.load(model_path)
    if getattr(bundle,'version',None) != path.name:
        raise RuntimeError(f'candidate bundle version mismatch: {path.name}')
    return bundle

def _metrics_for(bundle, csv_path: Path):
    df=pd.read_csv(csv_path)
    df['_normalized_url']=url_series(df)
    p=bundle.predict_proba(df['_normalized_url'].tolist())
    return evaluate(p,df['label'].astype(int).to_numpy(),bundle.threshold)


def _positive_feed_hits(bundle, csv_path: Path):
    df=pd.read_csv(csv_path)
    urls=url_series(df).tolist()
    if not urls: return None
    return {'hits': int(bundle.predict(urls).sum()), 'count': len(urls), 'recall': float(bundle.predict(urls).mean())}


def main():
    ap=argparse.ArgumentParser(description='Promote only a model that passes absolute gates and does not regress the current release on independent evidence.')
    ap.add_argument('version')
    ap.add_argument('--min-outer-accuracy',type=float,default=.98)
    ap.add_argument('--min-outer-precision',type=float,default=.98)
    ap.add_argument('--min-outer-recall',type=float,default=.97)
    ap.add_argument('--min-frozen-accuracy',type=float,default=.98)
    ap.add_argument('--min-frozen-precision',type=float,default=.99)
    ap.add_argument('--min-frozen-recall',type=float,default=.95)
    ap.add_argument('--allow-metric-tolerance',type=float,default=0.0)
    ap.add_argument('--realworld-positive',default='data/realworld/archive/openphish_20261005_unseen.csv')
    ap.add_argument('--legitimate-challenge',default='data/realworld/legitimate_challenge.csv')
    args=ap.parse_args()
    model_dir=ROOT/'models'/args.version
    # Accept candidates in models/candidates without treating them as rollback targets.
    if not model_dir.exists(): model_dir=ROOT/'models'/'candidates'/args.version
    metrics=json.loads((model_dir/'metrics.json').read_text())
    frozen=json.loads((model_dir/'frozen_regression_metrics.json').read_text())
    gates={
        'outer_accuracy':metrics['accuracy']>=args.min_outer_accuracy,
        'outer_precision':metrics['precision']>=args.min_outer_precision,
        'outer_recall':metrics['recall']>=args.min_outer_recall,
        'frozen_accuracy':frozen['accuracy']>=args.min_frozen_accuracy,
        'frozen_precision':frozen['precision']>=args.min_frozen_precision,
        'frozen_recall':frozen['recall']>=args.min_frozen_recall,
    }
    candidate=_load_candidate(model_dir) if model_dir.parent.name == 'candidates' else load_bundle(model_dir/'model.joblib')
    challenge_path=ROOT/args.legitimate_challenge
    challenge_result={'path':args.legitimate_challenge,'required_unique_domains':15}
    if challenge_path.exists():
        challenge_meta_path=challenge_path.with_suffix(challenge_path.suffix+'.meta.json')
        challenge_meta={}
        if challenge_meta_path.exists():
            challenge_meta=json.loads(challenge_meta_path.read_text(encoding='utf-8'))
        challenge_hash=hashlib.sha256(challenge_path.read_bytes()).hexdigest()
        challenge=pd.read_csv(challenge_path)
        file_integrity=(challenge_meta.get('sha256')==challenge_hash and challenge_meta.get('rows')==len(challenge) and bool(challenge_meta.get('label_source')))
        if not {'url','label'}.issubset(challenge.columns):
            raise ValueError('Legitimate challenge CSV must contain url,label columns')
        if not {'source','observed_at'}.issubset(challenge.columns):
            raise ValueError('Legitimate challenge rows must retain source and observation date')
        challenge['_normalized_url']=url_series(challenge)
        labels=pd.to_numeric(challenge['label'],errors='coerce').fillna(-1).astype(int).to_numpy()
        if not (labels == 0).all() or challenge['_normalized_url'].str.len().lt(8).any():
            raise ValueError('Legitimate challenge must contain valid URLs with label 0 only')
        unique_domains={registrable(u) for u in challenge['_normalized_url'].astype(str)}
        train_urls,train_domains=load_url_domain_sets(ROOT/'data/training/production_training.csv')
        challenge_urls=set(challenge['_normalized_url'].astype(str))
        duplicate_url_occurrences=len(challenge)-len(challenge_urls)
        exact_overlap=len(challenge_urls & train_urls)
        domain_overlap=len(unique_domains & train_domains)
        probs=candidate.predict_proba(challenge['_normalized_url'].tolist())
        predicted=(probs>=float(candidate.threshold)).astype(int)
        benign=labels==0
        fp=int(((predicted==1)&benign).sum())
        timestamp_coverage=float(challenge['observed_at'].fillna('').astype(str).str.strip().ne('').mean())
        challenge_result.update({'rows':len(challenge),'unique_urls':len(challenge_urls),'duplicate_url_occurrences':duplicate_url_occurrences,'unique_registrable_domains':len(unique_domains),'exact_training_overlap':exact_overlap,'domain_training_overlap':domain_overlap,'false_positives':fp,'false_positive_rate':float(fp/max(1,int(benign.sum()))),'threshold':float(candidate.threshold),'challenge_sha256':challenge_hash,'provenance_sha256_match':file_integrity,'observed_at_coverage':timestamp_coverage})
        gates['legitimate_challenge_zero_false_positives']=bool(benign.any() and fp==0)
        gates['legitimate_challenge_minimum_coverage']=len(unique_domains)>=15
        gates['legitimate_challenge_training_disjoint']=exact_overlap==0 and domain_overlap==0
        gates['legitimate_challenge_provenance']=file_integrity and bool(challenge_meta.get('label_source')) and timestamp_coverage>=0.8
        gates['legitimate_challenge_no_duplicate_urls']=duplicate_url_occurrences==0
    else:
        challenge_result['error']='challenge dataset missing'
        gates['legitimate_challenge_zero_false_positives']=False
        gates['legitimate_challenge_minimum_coverage']=False
        gates['legitimate_challenge_training_disjoint']=False
        gates['legitimate_challenge_provenance']=False
        gates['legitimate_challenge_no_duplicate_urls']=False
    cur=current(); comparison=None
    if cur and cur != args.version:
        current_model=load_bundle(ROOT/'models'/cur/'model.joblib')
        fr=ROOT/'data/frozen_regression.csv'
        if fr.exists():
            cfr=_metrics_for(candidate,fr); pfr=_metrics_for(current_model,fr)
            tol=float(args.allow_metric_tolerance)
            comparison={'frozen_candidate':cfr,'frozen_current':pfr,'frozen_non_regression':{
                'accuracy':cfr['accuracy']>=pfr['accuracy']-tol,
                'precision':cfr['precision']>=pfr['precision']-tol,
                'recall':cfr['recall']>=pfr['recall']-tol,
                'f1':cfr['f1']>=pfr['f1']-tol,
            }}
        rw=ROOT/args.realworld_positive
        if rw.exists():
            cp=_positive_feed_hits(candidate,rw); pp=_positive_feed_hits(current_model,rw)
            comparison['realworld_candidate']=cp;comparison['realworld_current']=pp
            comparison['realworld_non_regression']=cp['hits']>=pp['hits']
    if comparison:
        gates['non_regression_frozen']=all(comparison['frozen_non_regression'].values())
        if 'realworld_non_regression' in comparison: gates['non_regression_realworld']=comparison['realworld_non_regression']
    passed=all(gates.values())
    result={'version':args.version,'previous':cur,'passed':passed,'gates':gates,'legitimate_challenge':challenge_result,'comparison':comparison}
    if not passed:
        print(json.dumps(result,indent=2)); raise SystemExit(2)
    result['promotion']=promote(args.version) if (ROOT/'models'/args.version).exists() else {'error':'candidate promotion requires moving model into production registry'}
    if (ROOT/'models'/args.version).exists() and result['promotion'].get('current') == args.version:
        model_path=ROOT/'models'/args.version/'model.joblib'
        approval={
            'schema_version':1,'approved':True,'approved_at':datetime.now(timezone.utc).isoformat(),
            'model_version':args.version,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'challenge_sha256':challenge_result.get('provenance_sha256_match') and hashlib.sha256(challenge_path.read_bytes()).hexdigest(),
            'legitimate_challenge':challenge_result,'passed_gates':gates,
        }
        approval_path=ROOT/'models'/args.version/'BLOCKING_APPROVED.json'
        fd,tmp=tempfile.mkstemp(prefix='.BLOCKING_APPROVED.',dir=str(approval_path.parent),text=True)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as out:
                json.dump(approval,out,indent=2);out.flush();os.fsync(out.fileno())
            os.replace(tmp,approval_path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        result['blocking_approval_artifact']=str(approval_path.relative_to(ROOT))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
