# Kubernetes deployment

This is a production-oriented horizontal-scaling baseline. It does not claim a fixed RPS number; capacity must be established by a distributed k6 run against the actual cluster.

Build/push `phishguard:realtime`, create a Secret from `secret.example.yaml`, then apply:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/secret.example.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/pdb.yaml
```

Replace the example secret values before applying to any shared cluster. Run the distributed load test separately and record achieved RPS/p95/p99/error rate.

Each API pod shares an ephemeral feed directory with a restricted refresher
sidecar. It refreshes OpenPhish on a 12-hour cadence. URLhaus stays disabled
until its API key is provisioned. Feed refresh outcomes are visible from
`/health` without exposing provider error details or credentials. A pod
replacement starts with an empty feed volume and must refresh it again.

The sample deployment sets `ML_BLOCKING_ENABLED=false`. Keep that setting until
a candidate passes the real-world legitimate challenge promotion gates. The
included challenge currently has six named domains and intentionally fails the
minimum coverage requirement.
