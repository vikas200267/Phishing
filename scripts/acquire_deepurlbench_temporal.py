from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET = 'DeepInstinct/DeepURLBench'
ROWS_API = 'https://datasets-server.huggingface.co/rows'
TOTAL_ROWS = 40_194_045
PAGE_SIZE = 100


def fetch_page(offset: int, retries: int = 5) -> list[dict]:
    query = urllib.parse.urlencode({'dataset': DATASET, 'config': 'default', 'split': 'train', 'offset': offset, 'length': PAGE_SIZE})
    request = urllib.request.Request(f'{ROWS_API}?{query}', headers={'User-Agent': 'PhishGuard temporal benchmark acquisition (research)'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            result = []
            for item in payload.get('rows', []):
                row = item.get('row') or {}
                label = str(row.get('label') or '').lower()
                if label in {'benign', 'phishing'} and row.get('url') and row.get('first_seen'):
                    result.append({'url': str(row['url']).strip(), 'label': '1' if label == 'phishing' else '0', 'first_seen': str(row['first_seen']), 'source_row': str(item.get('row_idx', ''))})
            return result
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 10))
    return []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Acquire a reproducible sample from the timestamped DeepURLBench research dataset; never visit listed URLs.')
    parser.add_argument('--pages', type=int, default=300, help='Number of 100-row API pages to sample uniformly across the corpus.')
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data/realworld/external/deepurlbench_temporal_sample.csv')
    args = parser.parse_args()
    if args.pages < 1 or args.workers < 1:
        parser.error('--pages and --workers must be positive')

    page_count = (TOTAL_ROWS + PAGE_SIZE - 1) // PAGE_SIZE
    rng = random.Random(args.seed)
    offsets = sorted(rng.sample(range(page_count), min(args.pages, page_count)))
    offsets = [offset * PAGE_SIZE for offset in offsets]
    rows = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_page, offset): offset for offset in offsets}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                failures.append({'offset': futures[future], 'error': type(exc).__name__})
    if failures:
        raise RuntimeError(f'{len(failures)} dataset pages failed; refusing to publish a partial sample')
    # Exact URL duplicates with contradictory labels are removed as ambiguous.
    labels_by_url: dict[str, set[str]] = {}
    for row in rows:
        labels_by_url.setdefault(row['url'], set()).add(row['label'])
    conflicts = {url for url, labels in labels_by_url.items() if len(labels) > 1}
    seen = set()
    clean = []
    duplicate_urls = 0
    for row in sorted(rows, key=lambda item: (item['first_seen'], item['source_row'])):
        if row['url'] in conflicts:
            continue
        if row['url'] in seen:
            duplicate_urls += 1
            continue
        seen.add(row['url'])
        clean.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['url', 'label', 'first_seen', 'source_row'])
        writer.writeheader()
        writer.writerows(clean)
    from collections import Counter
    counts = Counter(row['label'] for row in clean)
    metadata = {
        'dataset': DATASET,
        'dataset_url': 'https://huggingface.co/datasets/DeepInstinct/DeepURLBench',
        'viewer_api': ROWS_API,
        'license': 'CC BY-NC 4.0; benchmark/research use only; do not use for commercial production training or redistribute the sample',
        'timestamp_column': 'first_seen',
        'timestamp_semantics': 'source dataset first-observed timestamp; timezone not specified in released schema',
        'label_mapping': {'0': 'benign', '1': 'phishing'},
        'source_total_rows': TOTAL_ROWS,
        'sample_pages_requested': len(offsets),
        'seed': args.seed,
        'sampled_source_row_count': len(rows),
        'rows_after_exact_deduplication': len(clean),
        'conflicting_url_groups_removed': len(conflicts),
        'duplicate_url_observations_removed': duplicate_urls,
        'label_counts': dict(counts),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'listed_urls_visited': False,
        'path': str(args.out),
        'sha256': file_sha256(args.out),
    }
    args.out.with_suffix(args.out.suffix + '.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
