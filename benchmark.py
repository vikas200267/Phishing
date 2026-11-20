from __future__ import annotations
import argparse, time
import pandas as pd
from pathlib import Path
from app.api import predict_urls
from app.train import url_series
from app.registry import current
from app.url_model import load_bundle

ROOT = Path(__file__).resolve().parent


def load_urls(path: str):
    df = pd.read_csv(path)
    if 'url' in df.columns:
        return [u for u in df['url'].dropna().astype(str).tolist() if u]
    return url_series(df).dropna().astype(str).tolist()


def main(path: str, batch: int = 1000):
    version = current()
    if not version:
        raise RuntimeError('No promoted model in models/CURRENT')
    model = load_bundle(ROOT / 'models' / version / 'model.joblib')
    urls = load_urls(path)
    total = 0
    started = time.perf_counter()
    lat = []
    for i in range(0, len(urls), batch):
        chunk = urls[i:i + batch]
        t = time.perf_counter()
        predict_urls(chunk, model)
        lat.append(time.perf_counter() - t)
        total += len(chunk)
    elapsed = time.perf_counter() - started
    print(f'benchmark_type=dataset_or_realworld_replay')
    print(f'model_version={version}')
    print(f'input={path}')
    print(f'predictions={total}')
    print(f'elapsed_seconds={elapsed:.4f}')
    print(f'predictions_per_second={total/elapsed:.2f}')
    print(f'avg_batch_ms={(sum(lat)/len(lat))*1000:.3f}')
    print('NOTE: replay benchmark only; not a production capacity claim.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(ROOT / 'data/realworld_load_corpus.json'))
    ap.add_argument('--batch', type=int, default=1000)
    a = ap.parse_args()
    # JSON replay corpus is handled directly; CSV datasets use load_urls().
    if a.input.endswith('.json'):
        import json
        payload = json.loads(Path(a.input).read_text(encoding='utf-8'))
        urls = [x.get('url') if isinstance(x, dict) else x for x in payload.get('urls', [])]
        tmp = ROOT / 'data' / '.benchmark_urls.csv'
        pd.DataFrame({'url': [u for u in urls if u]}).to_csv(tmp, index=False)
        try:
            main(str(tmp), a.batch)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        main(a.input, a.batch)
