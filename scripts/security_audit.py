from __future__ import annotations
import importlib.util, subprocess, sys


def main():
    if importlib.util.find_spec('pip_audit') is None:
        print('pip-audit is not installed in this environment; CVE status is UNVERIFIED.')
        print('Install it in an Internet-enabled build environment, then run: python -m pip_audit -r requirements.txt')
        raise SystemExit(2)
    raise SystemExit(subprocess.call([sys.executable,'-m','pip_audit','-r','requirements.txt']))

if __name__=='__main__': main()
