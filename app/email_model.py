from __future__ import annotations
import hashlib, joblib, numpy as np, pandas as pd
import re
from urllib.parse import urlparse
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from .email_features import extract_email_features

TEXT_FIELDS=['subject','body']
NUMERIC=['subject_len','body_len','url_count','unique_domain_count','html_ratio','digit_ratio','exclamation_count','urgency_hits','financial_hits','sender_reply_mismatch','sender_return_mismatch','spf_pass','dkim_pass','dmarc_pass','spf_fail','dkim_fail','dmarc_fail','has_attachment']

def email_text(row):
    sender_domain = row.get('sender_domain')
    if not isinstance(sender_domain, str) or not sender_domain.strip():
        sender = str(row.get('sender',''))
        sender_domain = sender.rsplit('@',1)[-1] if '@' in sender else ''
    return '\n'.join([
        'sender domain: '+str(sender_domain or ''),
        str(row.get('subject','')),
        str(row.get('body','')),
        'extracted URLs: '+str(row.get('urls','')),
    ])

def numeric_matrix(df):
    rows=[]
    for _,r in df.iterrows():
        features=extract_email_features(subject=r.get('subject',''),sender=r.get('sender',''),reply_to=r.get('reply_to',''),return_path=r.get('return_path',''),body=r.get('body',''),headers=r.get('headers') if isinstance(r.get('headers'),dict) else {})
        if 'url_count' in r and pd.notna(r.get('url_count')):
            features['url_count']=float(r.get('url_count'))
        if 'urls' in r and isinstance(r.get('urls'),str):
            domains=set()
            for url in re.findall(r'https?://[^\s<>"]+',r.get('urls',''),flags=re.I):
                try: domains.add(urlparse(url).hostname)
                except ValueError: continue
            features['unique_domain_count']=len({d.lower() for d in domains if d})
        if 'has_attachments' in r and pd.notna(r.get('has_attachments')):
            value=r.get('has_attachments')
            features['has_attachment']=int(str(value).strip().lower() in {'1','true','yes'} if isinstance(value,str) else bool(value))
        rows.append(features)
    return pd.DataFrame(rows,columns=NUMERIC).fillna(0).to_numpy(dtype=np.float32)

def train_email_csv(path: str, out_path: str, eval_path: str|None=None):
    df=pd.read_csv(path,low_memory=False)
    if 'label' not in df.columns: raise ValueError('Email CSV requires label 0/1')
    df=df[df.label.isin([0,1])].copy().reset_index(drop=True)
    y=df.label.astype(int).to_numpy()
    if len(df)<100 or len(np.unique(y))<2: raise ValueError('Need at least 100 labeled email rows containing both classes')
    if eval_path:
        train=df.reset_index(drop=True)
        test=pd.read_csv(eval_path,low_memory=False)
        test=test[test.label.isin([0,1])].copy().reset_index(drop=True)
        if len(test)<100 or test.label.astype(int).nunique()<2: raise ValueError('Temporal email evaluation file must contain >=100 rows and both classes')
        tr=np.arange(len(train)); te=None
    else:
        if 'sender_domain' in df:
            groups=df.sender_domain.fillna('').astype(str).str.lower()
        else:
            groups=df.get('sender',pd.Series('',index=df.index)).astype(str).str.extract(r'@([^> ]+)')[0].fillna('unknown').str.lower()
        gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
        tr,te=next(gss.split(df,y,groups=groups))
        train=df.iloc[tr].copy().reset_index(drop=True)
        test=df.iloc[te].copy().reset_index(drop=True)
    y_train=train.label.astype(int).to_numpy(); y_test=test.label.astype(int).to_numpy()
    vec=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True,strip_accents='unicode')
    Xtr=hstack([csr_matrix(numeric_matrix(train)),vec.fit_transform([email_text(r) for _,r in train.iterrows()])],format='csr')
    Xte=hstack([csr_matrix(numeric_matrix(test)),vec.transform([email_text(r) for _,r in test.iterrows()])],format='csr')
    base=LogisticRegression(max_iter=1000,class_weight='balanced',n_jobs=-1)
    model=CalibratedClassifierCV(base,method='sigmoid',cv=3); model.fit(Xtr,y[tr])
    prob=model.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    metrics={'train_samples':len(train),'test_samples':len(test),'threshold':.5,
        'accuracy':float(accuracy_score(y_test,pred)),'precision':float(precision_score(y_test,pred,zero_division=0)),
        'recall':float(recall_score(y_test,pred,zero_division=0)),'f1':float(f1_score(y_test,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(y_test,prob)),'pr_auc':float(average_precision_score(y_test,prob)),
        'brier_score':float(brier_score_loss(y_test,prob)),'false_positive_rate':float(fp/max(1,fp+tn)),
        'false_negative_rate':float(fn/max(1,fn+tp)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
        'train_label_counts':{str(k):int(v) for k,v in train.label.value_counts().sort_index().items()},
        'test_label_counts':{str(k):int(v) for k,v in test.label.value_counts().sort_index().items()},
        'temporal_evaluation':bool(eval_path)}
    if 'observed_at' in test:
        ts=pd.to_datetime(test.observed_at,errors='coerce',utc=True).dropna()
        metrics.update({'observation_time_coverage':float(test.observed_at.fillna('').astype(str).str.strip().ne('').mean()),'test_time_start':ts.min().isoformat() if len(ts) else None,'test_time_end':ts.max().isoformat() if len(ts) else None})
    if 'sender_domain' in train and 'sender_domain' in test:
        train_domains=set(train.sender_domain.fillna('').astype(str).str.lower())
        test_domains=set(test.sender_domain.fillna('').astype(str).str.lower())
        metrics['sender_domain_overlap_count']=len((train_domains & test_domains)-{''})
        metrics['unique_train_sender_domains']=len(train_domains-{''})
        metrics['unique_test_sender_domains']=len(test_domains-{''})
    metrics['train_source_counts']={str(k):int(v) for k,v in train.get('source',pd.Series('unknown',index=train.index)).fillna('unknown').value_counts().sort_index().items()}
    metrics['test_source_counts']={str(k):int(v) for k,v in test.get('source',pd.Series('unknown',index=test.index)).fillna('unknown').value_counts().sort_index().items()}
    bundle={'model':model,'vectorizer':vec,'numeric':NUMERIC,'metrics':metrics,'text_schema':'sender_domain, subject, body, extracted URLs; numeric email and source URL-count features'}
    joblib.dump(bundle,out_path,compress=3)
    if eval_path:
        pd.DataFrame({'label':y_test,'risk_score':prob,'prediction':pred,'source':test.get('source','unknown'),'observed_at':test.get('observed_at','')}).to_csv(str(out_path)+'.predictions.csv',index=False)
    import json
    from pathlib import Path
    Path(str(out_path)+'.metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics

def predict_email(bundle, payloads):
    df=pd.DataFrame(payloads)
    X=hstack([csr_matrix(numeric_matrix(df)),bundle['vectorizer'].transform([email_text(r) for _,r in df.iterrows()])],format='csr')
    return bundle['model'].predict_proba(X)[:,1]
