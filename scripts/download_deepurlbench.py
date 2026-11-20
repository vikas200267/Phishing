from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FILES = {
    'deepurlbench_0000.parquet': ('https://huggingface.co/datasets/DeepInstinct/DeepURLBench/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet', 'de344c0b7000483cca57fcbeb11a727dafe40fa670220e17567f008c5b120ff7', 942_899_523),
    'deepurlbench_0001.parquet': ('https://huggingface.co/datasets/DeepInstinct/DeepURLBench/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet', 'f39c11f915169accf5bd8e553026cd006f7b93de7da0f914a36ee309f7f4dfce', 1_094_091_115),
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Download the public DeepURLBench temporal research corpus with checksums; never visit URLs in the dataset.')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data/realworld/external/deepurlbench')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {'dataset': 'DeepInstinct/DeepURLBench', 'source_url': 'https://huggingface.co/datasets/DeepInstinct/DeepURLBench', 'license': 'CC BY-NC 4.0; noncommercial research/evaluation only; do not use for commercial production training or redistribution', 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'files': [], 'listed_urls_visited': False}
    for name, (url, expected_sha, expected_size) in FILES.items():
        target = args.out_dir / name
        digest = hashlib.sha256()
        size = 0
        if target.exists():
            existing_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if target.stat().st_size == expected_size and existing_hash == expected_sha:
                report['files'].append({'path': str(target), 'bytes': expected_size, 'sha256': expected_sha, 'reused_verified_file': True})
                continue
        fd, tmp = tempfile.mkstemp(prefix=f'.{name}.', suffix='.partial', dir=args.out_dir)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'PhishGuard temporal benchmark acquisition (research)'})
            with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(request, timeout=180) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                    if size % (100 * 1024 * 1024) < 1024 * 1024:
                        print(f'{name}: {size}/{expected_size} bytes', flush=True)
                output.flush()
                os.fsync(output.fileno())
            actual = digest.hexdigest()
            if size != expected_size or actual != expected_sha:
                raise RuntimeError(f'{name} checksum/size mismatch: bytes={size}, sha256={actual}')
            os.replace(tmp, target)
            report['files'].append({'path': str(target), 'bytes': size, 'sha256': actual, 'reused_verified_file': False})
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    report_path = args.out_dir / 'download_manifest.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
