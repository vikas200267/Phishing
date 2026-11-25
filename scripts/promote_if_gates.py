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
import numpy as np
import joblib
from app.registry import promote, current
from app.domain import DOMAIN_GROUPING_VERSION
from app.url_model import load_bundle
from app.train import domain_groups, evaluate, url_series
from scripts.evaluate_phishvn_temporal import score_file
from scripts.model_temporal_evidence import verify_model_temporal_inputs
from scripts.evaluate_shortcuts import scheme_class_counts

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


def _calibration_metrics(path: Path, expected_hash: str | None, threshold: float):
    if not path.is_file() or not expected_hash or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        return None
    frame=pd.read_csv(path, float_precision='round_trip')
    if not {'label','probability'}.issubset(frame.columns) or frame.empty:
        return None
    labels=pd.to_numeric(frame['label'], errors='coerce').to_numpy()
    probabilities=pd.to_numeric(frame['probability'], errors='coerce').to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        return None
    return evaluate(probabilities, labels.astype(int), float(threshold))


def main():
    ap=argparse.ArgumentParser(description='Promote only a model that passes absolute gates and does not regress the current release on independent evidence.')
    ap.add_argument('version')
    ap.add_argument('--min-outer-accuracy',type=float,default=.98)
    ap.add_argument('--min-outer-precision',type=float,default=.98)
    ap.add_argument('--min-outer-recall',type=float,default=.97)
    ap.add_argument('--min-frozen-accuracy',type=float,default=.98)
    ap.add_argument('--min-frozen-precision',type=float,default=.99)
    ap.add_argument('--min-frozen-recall',type=float,default=.98)
    ap.add_argument('--min-temporal-precision',type=float,default=.98)
    ap.add_argument('--min-temporal-recall',type=float,default=.97)
    ap.add_argument('--max-temporal-fpr',type=float,default=.02)
    ap.add_argument('--allow-metric-tolerance',type=float,default=0.0)
    ap.add_argument('--realworld-positive',default='data/realworld/archive/openphish_20261005_unseen.csv')
    ap.add_argument('--legitimate-challenge',default='data/realworld/legitimate_challenge.csv')
    args=ap.parse_args()
    model_dir=ROOT/'models'/args.version
    # Accept candidates in models/candidates without treating them as rollback targets.
    if not model_dir.exists(): model_dir=ROOT/'models'/'candidates'/args.version
    metadata=json.loads((model_dir/'metadata.json').read_text())
    recorded_metrics=json.loads((model_dir/'metrics.json').read_text())
    frozen_path=ROOT/'data/frozen_regression.csv'
    candidate=_load_candidate(model_dir) if model_dir.parent.name == 'candidates' else load_bundle(model_dir/'model.joblib')
    frozen=_metrics_for(candidate,frozen_path)
    validation_path=model_dir/'validation_predictions.csv'
    validation_hash=hashlib.sha256(validation_path.read_bytes()).hexdigest() if validation_path.exists() else None
    validation_ok=bool(validation_hash and validation_hash == metadata.get('validation_predictions_sha256'))
    metrics=_metrics_for(candidate,validation_path) if validation_ok else {'accuracy':0.0,'precision':0.0,'recall':0.0}
    gates={
        'outer_accuracy':metrics['accuracy']>=args.min_outer_accuracy,
        'outer_precision':metrics['precision']>=args.min_outer_precision,
        'outer_recall':metrics['recall']>=args.min_outer_recall,
        'frozen_accuracy':frozen['accuracy']>=args.min_frozen_accuracy,
        'frozen_precision':frozen['precision']>=args.min_frozen_precision,
        'frozen_recall':frozen['recall']>=args.min_frozen_recall,
        'release_thresholds_enforced': (
            args.min_outer_accuracy >= .98 and args.min_outer_precision >= .98 and args.min_outer_recall >= .97
            and args.min_frozen_accuracy >= .98 and args.min_frozen_precision >= .99 and args.min_frozen_recall >= .98
            and args.min_temporal_precision >= .98 and args.min_temporal_recall >= .97
            and 0 <= args.max_temporal_fpr <= .02 and args.allow_metric_tolerance == 0
        ),
    }
    gates['validation_predictions_integrity'] = validation_ok
    gates['independent_calibration_protocol'] = metadata.get('stacking_split_protocol') == 'domain_disjoint_fit_calibration_threshold_v1'
    calibration_path=model_dir/'calibration_scores.csv'
    calibration_metrics=_calibration_metrics(calibration_path,metadata.get('calibration_scores_sha256'),float(candidate.threshold))
    gates['calibration_scores_integrity']=calibration_metrics is not None
    gates['calibration_precision']=bool(calibration_metrics and calibration_metrics['precision'] >= .985)
    gates['calibration_recall']=bool(calibration_metrics and calibration_metrics['recall'] >= .95)
    gates['calibration_false_positive_rate']=bool(calibration_metrics and calibration_metrics['false_positive_rate'] <= .01)
    gates['calibration_support']=bool(calibration_metrics and calibration_metrics['tp'] + calibration_metrics['fn'] >= 100 and calibration_metrics['tn'] + calibration_metrics['fp'] >= 100)
    gates['domain_grouping_consistent'] = metadata.get('domain_grouping_version') == DOMAIN_GROUPING_VERSION
    training_domains_path=model_dir/'training_domains.txt'
    training_hash=hashlib.sha256(training_domains_path.read_bytes()).hexdigest() if training_domains_path.exists() else None
    training_domains_ok=bool(training_hash and training_hash == metadata.get('training_domains_sha256'))
    if validation_ok and training_domains_ok:
        validation=pd.read_csv(validation_path)
        validation['_normalized_url']=url_series(validation)
        validation_domains=set(domain_groups(validation).astype(str))
        training_domains=set(training_domains_path.read_text(encoding='utf-8').splitlines())
        gates['outer_data_independent']=bool(gates['domain_grouping_consistent'] and validation_domains and validation_domains.isdisjoint(training_domains))
    else:
        gates['outer_data_independent']=False
    prepared_domains_path=model_dir/'prepared_domains.txt'
    prepared_hash=hashlib.sha256(prepared_domains_path.read_bytes()).hexdigest() if prepared_domains_path.exists() else None
    prepared_domains_ok=bool(gates['domain_grouping_consistent'] and prepared_hash and prepared_hash == metadata.get('prepared_domains_sha256'))
    candidate_domains=set(prepared_domains_path.read_text(encoding='utf-8').splitlines()) if prepared_domains_ok else set()
    gates['prepared_domains_integrity']=prepared_domains_ok
    rw_path=ROOT/args.realworld_positive
    gates['realworld_sample_is_default']=rw_path.resolve()==(ROOT/'data/realworld/archive/openphish_20261005_unseen.csv').resolve()
    if rw_path.is_file() and prepared_domains_ok:
        rw_frame=pd.read_csv(rw_path)
        rw_frame['_normalized_url']=url_series(rw_frame)
        rw_domains=set(domain_groups(rw_frame).astype(str))
        gates['realworld_domain_disjoint']=bool(len(rw_frame)>=30 and rw_domains and rw_domains.isdisjoint(candidate_domains))
    else:
        gates['realworld_domain_disjoint']=False
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
        unique_domains=set(domain_groups(challenge).astype(str))
        challenge_urls=set(challenge['_normalized_url'].astype(str))
        duplicate_url_occurrences=len(challenge)-len(challenge_urls)
        domain_overlap=len(unique_domains & candidate_domains) if prepared_domains_ok else None
        # Exact URL overlap is impossible when the recorded prepared domain
        # groups are complete and disjoint. Otherwise it is unknown, not zero.
        exact_overlap=0 if prepared_domains_ok and domain_overlap == 0 else None
        probs=candidate.predict_proba(challenge['_normalized_url'].tolist())
        predicted=(probs>=float(candidate.threshold)).astype(int)
        benign=labels==0
        fp=int(((predicted==1)&benign).sum())
        timestamp_coverage=float(challenge['observed_at'].fillna('').astype(str).str.strip().ne('').mean())
        verified = challenge.get('verification_status', pd.Series('', index=challenge.index)).fillna('').eq('verified_http_html')
        verified &= pd.to_datetime(challenge['observed_at'], errors='coerce', utc=True).notna()
        verified_domains=set(domain_groups(challenge.loc[verified]).astype(str)) - {''}
        challenge_result.update({'rows':len(challenge),'unique_urls':len(challenge_urls),'duplicate_url_occurrences':duplicate_url_occurrences,'unique_registrable_domains':len(unique_domains),'verified_unique_domains':len(verified_domains),'exact_training_overlap':exact_overlap,'domain_training_overlap':domain_overlap,'false_positives':fp,'false_positive_rate':float(fp/max(1,int(benign.sum()))),'threshold':float(candidate.threshold),'challenge_sha256':challenge_hash,'provenance_sha256_match':file_integrity,'observed_at_coverage':timestamp_coverage})
        gates['legitimate_challenge_zero_false_positives']=bool(benign.any() and fp==0)
        gates['legitimate_challenge_minimum_coverage']=len(unique_domains)>=15
        gates['legitimate_challenge_verified_coverage']=len(verified_domains)>=15
        gates['legitimate_challenge_training_disjoint']=prepared_domains_ok and exact_overlap==0 and domain_overlap==0
        gates['legitimate_challenge_is_default']=challenge_path.resolve()==(ROOT/'data/realworld/legitimate_challenge.csv').resolve()
        gates['legitimate_challenge_provenance']=file_integrity and bool(challenge_meta.get('label_source')) and timestamp_coverage>=0.8
        gates['legitimate_challenge_no_duplicate_urls']=duplicate_url_occurrences==0
    else:
        challenge_result['error']='challenge dataset missing'
        gates['legitimate_challenge_zero_false_positives']=False
        gates['legitimate_challenge_minimum_coverage']=False
        gates['legitimate_challenge_verified_coverage']=False
        gates['legitimate_challenge_training_disjoint']=False
        gates['legitimate_challenge_provenance']=False
        gates['legitimate_challenge_no_duplicate_urls']=False
        gates['legitimate_challenge_is_default']=False
    temporal_evidence=verify_model_temporal_inputs(model_dir)
    temporal_metrics=None
    temporal_scheme_counts=None
    if temporal_evidence['qualified']:
        temporal_path=ROOT/'data/realworld/evaluations/phishvn_temporal/test.csv'
        temporal_metrics=score_file(candidate, temporal_path)
        temporal_scheme_counts=scheme_class_counts(pd.read_csv(temporal_path))
    gates['temporal_model_inputs_qualified']=temporal_evidence['qualified'] is True
    gates['temporal_precision']=bool(temporal_metrics and temporal_metrics['precision'] >= args.min_temporal_precision)
    gates['temporal_recall']=bool(temporal_metrics and temporal_metrics['recall'] >= args.min_temporal_recall)
    gates['temporal_false_positive_rate']=bool(temporal_metrics and temporal_metrics['false_positive_rate'] <= args.max_temporal_fpr)
    gates['temporal_support']=bool(temporal_metrics and temporal_metrics['tp'] + temporal_metrics['fn'] >= 50 and temporal_metrics['tn'] + temporal_metrics['fp'] >= 50)
    gates['temporal_scheme_class_support']=bool(temporal_scheme_counts and all(n >= 10 for n in temporal_scheme_counts.values()))
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
    gates.setdefault('non_regression_frozen',False)
    gates.setdefault('non_regression_realworld',False)
    passed=all(gates.values())
    result={'version':args.version,'previous':cur,'passed':passed,'gates':gates,'calibration_metrics':calibration_metrics,'outer_metrics':metrics,'recorded_outer_metrics':recorded_metrics,'frozen_metrics':frozen,'legitimate_challenge':challenge_result,'temporal_evidence':temporal_evidence,'temporal_metrics':temporal_metrics,'temporal_scheme_class_counts':temporal_scheme_counts,'comparison':comparison}
    if not passed:
        print(json.dumps(result,indent=2)); raise SystemExit(2)
    if (ROOT/'models'/args.version).exists():
        model_path=ROOT/'models'/args.version/'model.joblib'
        approval={
            'schema_version':2,'approved':True,'approved_at':datetime.now(timezone.utc).isoformat(),
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
        result['promotion']=promote(args.version)
    else:
        result['promotion']={'error':'candidate promotion requires moving model into production registry'}
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
