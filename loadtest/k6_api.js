import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

const base = __ENV.BASE_URL || 'http://localhost:8000';
const key = __ENV.PHISHGUARD_API_KEY || '';
const target = Number(__ENV.TARGET_RPS || 1000);
const duration = __ENV.DURATION || '60s';
const batchSize = Number(__ENV.BATCH_SIZE || 1);
const mode = __ENV.MODE || 'inference';
const corpusPath = __ENV.CORPUS_PATH || '/data/live_load_corpus.json';
const corpusPayload = JSON.parse(open(corpusPath));
const corpus = (corpusPayload.urls || []).map((x) => x.url || x).filter(Boolean);
if (!corpus.length) throw new Error('Real-world corpus is empty. Run scripts/build_realworld_corpus.py first.');
const errors = new Rate('phishguard_errors');

function corpusUrl(i) { return corpus[i % corpus.length]; }

export const options = {
  discardResponseBodies: true,
  scenarios: {
    load: {
      executor: 'constant-arrival-rate',
      rate: target,
      timeUnit: '1s',
      duration,
      preAllocatedVUs: Number(__ENV.PREALLOCATED_VUS || Math.max(100, Math.ceil(target / 50))),
      maxVUs: Number(__ENV.MAX_VUS || Math.max(1000, Math.ceil(target / 10))),
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<250', 'p(99)<500'],
    phishguard_errors: ['rate<0.01'],
  },
};

export default function () {
  const seed = (__ITER + __VU) % corpus.length;
  let path, payload;
  if (mode === 'batch' && batchSize > 1) {
    path = '/v1/predict/batch';
    const urls = [];
    for (let i = 0; i < batchSize; i++) urls.push(corpusUrl(seed + i));
    payload = JSON.stringify({ urls });
  } else {
    path = '/v1/predict';
    payload = JSON.stringify({ url: corpusUrl(seed), enrich: false });
  }
  const res = http.post(`${base}${path}`, payload, { headers: { 'Content-Type': 'application/json', 'X-API-Key': key } });
  const ok = check(res, { 'status 200': r => r.status === 200 });
  errors.add(!ok);
}
