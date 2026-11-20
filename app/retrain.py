from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from .feedback import verified_rows
from .train import train, build_arg_parser

ROOT=Path(__file__).resolve().parents[1]


def feedback_frame(limit=100000):
    rows=[]
    for u,label in verified_rows(limit):
        p=urlparse(u)
        rows.append({'url':u,'protocol':p.scheme,'domain_name':p.hostname or '', 'address':p.path + (('?' + p.query) if p.query else ''), 'label':int(label)})
    return pd.DataFrame(rows).drop_duplicates(subset=['url','label'])


def build_training_namespace(a, tmp: Path):
    parser=build_arg_parser()
    cli=[
        '--phishing',a.phishing,'--legitimate',a.legitimate,'--extra',str(tmp),
        '--trees',str(a.trees),'--depth',str(a.depth),'--lr',str(a.lr),
        '--lgb-trees',str(a.lgb_trees),'--lgb-leaves',str(a.lgb_leaves),'--lgb-lr',str(a.lgb_lr),
        '--text-C',str(a.text_C),'--char-features',str(a.char_features),'--word-features',str(a.word_features),
        '--threads',str(a.threads),
        '--folds',str(a.folds),
        '--min-precision',str(a.min_precision),'--min-recall',str(a.min_recall),
        '--threshold-min',str(a.threshold_min),'--threshold-max',str(a.threshold_max),
    ]
    if a.promote: cli.append('--promote')
    return parser.parse_args(cli)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Retrain PhishGuard only from verified feedback plus immutable base data.')
    ap.add_argument('--phishing',required=True); ap.add_argument('--legitimate',required=True); ap.add_argument('--min-feedback',type=int,default=25)
    ap.add_argument('--trees',type=int,default=250); ap.add_argument('--depth',type=int,default=5); ap.add_argument('--lr',type=float,default=.035)
    ap.add_argument('--lgb-trees',type=int,default=220); ap.add_argument('--lgb-leaves',type=int,default=24); ap.add_argument('--lgb-lr',type=float,default=.035)
    ap.add_argument('--text-C',type=float,default=4.0); ap.add_argument('--char-features',type=int,default=25000); ap.add_argument('--word-features',type=int,default=12000); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--folds',type=int,default=3); ap.add_argument('--min-precision',type=float,default=.98); ap.add_argument('--min-recall',type=float,default=.95); ap.add_argument('--threshold-min',type=float,default=.12); ap.add_argument('--threshold-max',type=float,default=.45)
    ap.add_argument('--promote',action='store_true')
    a=ap.parse_args(); fb=feedback_frame()
    if len(fb)<a.min_feedback: raise SystemExit(f'Only {len(fb)} verified feedback rows; need at least {a.min_feedback}.')
    tmp=ROOT/'data'/'verified_feedback.csv'; fb.to_csv(tmp,index=False)
    train(build_training_namespace(a,tmp))
