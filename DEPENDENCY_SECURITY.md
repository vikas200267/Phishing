# Dependency security status

## Current artifact

The currently bundled model was trained under scikit-learn 1.8.0 in the isolated build runtime. The project now includes a CI `pip-audit` gate and `requirements-audit.txt` so an Internet-enabled build can produce a reproducible vulnerability report.

Run:

```bash
python -m pip install -r requirements.txt -r requirements-audit.txt
python -m pip_audit -r requirements.txt
```

The CI workflow runs this automatically on pushes and pull requests.

## Secure rebuild

The current official scikit-learn documentation identifies 1.9.1 as the current stable release. The file `requirements.secure-build.txt` requests `scikit-learn>=1.9.1,<2` for the **next clean retrain/rebuild**. Do not silently swap the dependency under the existing serialized model; retrain and run the complete model/regression gate first.

No statement of "zero CVEs" is made until `pip-audit` actually runs against the resolved dependency set.
