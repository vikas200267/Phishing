"""Build a reproducible set of evaluation URLs to exclude by domain."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from app.train import url_series


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    urls = set()
    inputs = []
    for name in args.inputs:
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f'Required evaluation dataset missing: {path}')
        frame = pd.read_csv(path, low_memory=False)
        values = url_series(frame).dropna().astype(str)
        urls.update(values)
        inputs.append({'path': name, 'sha256': sha256(path), 'rows': len(frame)})
    if not urls:
        raise ValueError('No evaluation URLs to reserve')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'url': sorted(urls)}).to_csv(args.out, index=False)
    manifest = {'generated_at': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
                'output_rows': len(urls), 'output_sha256': sha256(args.out),
                'usage': 'Pass this file to --exclude-domains-csv before model training.'}
    args.out.with_suffix(args.out.suffix + '.meta.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
