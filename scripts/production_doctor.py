from __future__ import annotations
import ast, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROD_FILES=[ROOT/'app'/'api.py',ROOT/'app'/'url_model.py',ROOT/'app'/'local_intel.py',ROOT/'app'/'registry.py',ROOT/'app'/'features.py',ROOT/'app'/'policy.py']
FORBIDDEN_PATTERNS=[
    r'\bRandomForestClassifier\s*\(',
    r'\bExtraTreesClassifier\s*\([^)]*synthetic',
    r'\.fit\s*\(\s*(?:synthetic|fake|dummy)_',
    r'benign_seed\s*\.\s*append',
    r'\bEXAMPLE_URLS\s*=',
]

def main():
    issues=[]
    for p in PROD_FILES:
        text=p.read_text(encoding='utf-8')
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat,text,re.I|re.S): issues.append({'file':str(p),'pattern':pat})
        try: ast.parse(text)
        except SyntaxError as e: issues.append({'file':str(p),'syntax_error':str(e)})
    # Require explicit fail-closed model loading and HTTP-only validation.
    api=(ROOT/'app'/'api.py').read_text(encoding='utf-8')
    model=(ROOT/'app'/'url_model.py').read_text(encoding='utf-8')
    policy=(ROOT/'app'/'policy.py').read_text(encoding='utf-8')
    checks={'model_load_fail_closed': 'load_bundle' in api and 'load_current' in api,
            'integrity_verification': 'verify_model_integrity' in model,
            'http_validation': "{'http', 'https'}" in policy,
            'production_doctor_pass': not issues}
    print(json.dumps({'checks':checks,'issues':issues},indent=2)); raise SystemExit(1 if issues or not all(checks.values()) else 0)
if __name__=='__main__':main()
