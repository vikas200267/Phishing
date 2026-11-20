from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd
from .features import FEATURE_NAMES, dataframe_to_features

EPS=1e-6

def _bin_edges(s: pd.Series, bins: int = 10):
    q=s.quantile([i/bins for i in range(bins+1)]).drop_duplicates().tolist()
    if len(q)<3: return None
    q[0], q[-1] = -math.inf, math.inf
    return q

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges=_bin_edges(expected,bins)
    if not edges: return 0.0
    e=pd.cut(expected, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    a=pd.cut(actual, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
    total=0.0
    for pe, pa in zip(e.tolist(), a.tolist()):
        pe=max(float(pe),EPS); pa=max(float(pa),EPS); total += (pa-pe)*math.log(pa/pe)
    return float(total)

def dataset_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    xb=dataframe_to_features(baseline_df); xc=dataframe_to_features(current_df)
    rows=[]
    for c in FEATURE_NAMES:
        score=psi(xb[c], xc[c])
        rows.append((c,score,"high" if score>=0.25 else "medium" if score>=0.10 else "low"))
    rows.sort(key=lambda x:x[1],reverse=True)
    return {"max_psi":rows[0][1] if rows else 0.0,"high_drift_features":[r for r in rows if r[2]=="high"],"all_features":rows}

def save_baseline(path: str, df: pd.DataFrame):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); dataframe_to_features(df).to_parquet(p,index=False)

def compare_to_baseline(path: str, current_df: pd.DataFrame) -> dict:
    base=pd.read_parquet(path); cur=dataframe_to_features(current_df)
    rows=[]
    for c in FEATURE_NAMES:
        score=psi(base[c],cur[c]); rows.append((c,score))
    rows.sort(key=lambda x:x[1],reverse=True)
    return {"max_psi":rows[0][1] if rows else 0.0,"top_features":rows[:10]}
