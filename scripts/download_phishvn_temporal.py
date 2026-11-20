from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = 'b97hxbxtpd'
VERSION = 5
FILENAME = 'PhishVN_v3.1.1_open.zip'
EXPECTED_SIZE = 6_697_259
EXPECTED_SHA256 = '4c610431fd275d8348695653eeacdece861dc7dc65ab09a2068d519247988968'
OUTPUT = Path(__file__).resolve().parents[1] / 'data/realworld/external/phishvn'


def main() -> int:
    listing_url = f'https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?folder_id=root&version={VERSION}'
    request = urllib.request.Request(listing_url, headers={'User-Agent': 'PhishGuard temporal evaluation data acquisition'})
    with urllib.request.urlopen(request, timeout=45) as response:
        files = json.load(response)
    record = next((item for item in files if item.get('filename') == FILENAME), None)
    if not record:
        raise RuntimeError(f'Could not find {FILENAME} in the Mendeley public file manifest.')
    if record.get('size') != EXPECTED_SIZE or record.get('content_details', {}).get('sha256_hash') != EXPECTED_SHA256:
        raise RuntimeError('Mendeley file metadata changed; inspect version and checksum before downloading.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / FILENAME
    fd, temp_path = tempfile.mkstemp(prefix=f'.{FILENAME}.', suffix='.partial', dir=OUTPUT)
    digest = hashlib.sha256()
    size = 0
    try:
        download = urllib.request.Request(record['content_details']['download_url'], headers={'User-Agent': 'PhishGuard temporal evaluation data acquisition'})
        with os.fdopen(fd, 'wb') as out, urllib.request.urlopen(download, timeout=120) as source:
            while block := source.read(1024 * 1024):
                out.write(block)
                digest.update(block)
                size += len(block)
            out.flush()
            os.fsync(out.fileno())
        if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError(f'Checksum mismatch: bytes={size}, sha256={digest.hexdigest()}')
        os.replace(temp_path, archive)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (OUTPUT / member.filename).resolve()
            if OUTPUT.resolve() not in target.parents and target != OUTPUT.resolve():
                raise RuntimeError(f'Unsafe archive path: {member.filename}')
        zf.extractall(OUTPUT)
    manifest = {
        'dataset': 'PhishVN: A Time-Stamped Vietnamese URL Phishing Dataset with Impersonation-Scenario Labels and Confidence Tiers',
        'dataset_url': f'https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}',
        'doi': f'10.17632/{DATASET_ID}.{VERSION}',
        'version': VERSION,
        'license': 'CC BY 4.0',
        'attribution': 'Thai Nguyen Vu; upstream sources: NCSC Tin Nhiem Mang, ChongLuaDao, and Tranco',
        'archive': str(archive),
        'archive_bytes': size,
        'archive_sha256': digest.hexdigest(),
        'file_manifest': record,
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'listed_urls_visited': False,
        'timestamp_semantics': 'collected_at: source-attested detection or certification date; scraped_at is retrieval time and is never used for temporal splitting',
    }
    (OUTPUT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
