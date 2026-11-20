from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

URL = 'https://zenodo.org/records/18471483/files/meajor_cleaned_preprocessed.csv?download=1'
EXPECTED_MD5 = 'aa8f59e96787cbd696c0b650e5400dc9'


def main() -> int:
    parser = argparse.ArgumentParser(description='Download the public MeAJOR email corpus and verify its Zenodo MD5 checksum.')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data/realworld/external/meajor_cleaned_preprocessed.csv')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{args.out.name}.', dir=args.out.parent)
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'PhishGuard research data acquisition'})
        with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(request, timeout=180) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                md5.update(chunk)
                sha256.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if md5.hexdigest() != EXPECTED_MD5:
            raise RuntimeError(f'Zenodo MD5 mismatch: expected {EXPECTED_MD5}, got {md5.hexdigest()}')
        os.replace(temporary, args.out)
        print({'path': str(args.out), 'bytes': args.out.stat().st_size, 'md5': md5.hexdigest(), 'sha256': sha256.hexdigest(), 'source': 'https://zenodo.org/records/18471483'})
        return 0
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
