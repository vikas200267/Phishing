from __future__ import annotations

import argparse
import asyncio
import json
import time
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def load_corpus(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding='utf-8'))
    urls = [x.get('url') if isinstance(x, dict) else x for x in data.get('urls', [])]
    urls = [str(u) for u in urls if u]
    if not urls:
        raise ValueError('real-world corpus contains no URLs')
    return urls


async def main_async(args):
    expected = args.expected_model_version
    urls = load_corpus(Path(args.corpus))
    sem = asyncio.Semaphore(args.concurrency)
    headers = {'X-API-Key': args.api_key, 'Content-Type': 'application/json'}
    success = 0; fail = 0; latencies = []
    start = time.perf_counter()

    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=args.timeout, limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)) as client:
        async def one(i: int):
            nonlocal success, fail
            async with sem:
                t = time.perf_counter()
                try:
                    if args.batch_size > 1:
                        payload_urls = [urls[(i * args.batch_size + j) % len(urls)] for j in range(args.batch_size)]
                        r = await client.post('/v1/predict/batch', json={'urls': payload_urls})
                    else:
                        r = await client.post('/v1/predict', json={'url': urls[i % len(urls)], 'enrich': False})
                    latencies.append(time.perf_counter() - t)
                    if r.status_code == 200:
                        try:
                            body = r.json()
                            if expected and body.get('model_version') != expected:
                                fail += 1
                            else:
                                success += 1
                        except Exception:
                            fail += 1
                    else:
                        fail += 1
                except Exception:
                    latencies.append(time.perf_counter() - t); fail += 1

        await asyncio.gather(*(one(i) for i in range(args.requests)))
    elapsed = time.perf_counter() - start
    latencies.sort()
    p95 = latencies[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None
    p99 = latencies[min(len(latencies)-1, int(len(latencies)*.99))] if latencies else None
    decisions = success * max(1, args.batch_size)
    result = {
        'mode': 'real_world_feed_replay_http',
        'corpus': str(args.corpus),
        'corpus_count': len(urls),
        'requests': args.requests,
        'concurrency': args.concurrency,
        'successful': success,
        'failed': fail,
        'elapsed_seconds': elapsed,
        'requests_per_second': args.requests / elapsed if elapsed else None,
        'batch_size': args.batch_size,
        'decisions': decisions,
        'decisions_per_second': decisions / elapsed if elapsed else None,
        'p95_ms': p95 * 1000 if p95 is not None else None,
        'p99_ms': p99 * 1000 if p99 is not None else None,
        'cache': 'disabled',
        'local_intel': 'disabled',
        'cascade': 'enabled' if os.getenv('CASCADE_ENABLED','false').lower() == 'true' else 'disabled',
        'expected_model_version': expected,
        'listed_urls_visited': False,
    }
    print(json.dumps(result, indent=2))
    return 0 if fail == 0 else 1


def main():
    p = argparse.ArgumentParser(description='Benchmark the real HTTP inference path using captured real-world feed URLs only.')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--api-key', required=True)
    p.add_argument('--corpus', default='data/realworld_load_corpus.json')
    p.add_argument('--requests', type=int, default=1000)
    p.add_argument('--concurrency', type=int, default=50)
    p.add_argument('--timeout', type=float, default=10.0)
    p.add_argument('--expected-model-version', default=os.getenv('EXPECTED_MODEL_VERSION',''))
    p.add_argument('--batch-size', type=int, default=1)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))

if __name__ == '__main__': main()
