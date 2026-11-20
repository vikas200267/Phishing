from __future__ import annotations
import json, sqlite3, hashlib, time
from pathlib import Path
from .features import normalize_url

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; DB=Path(__import__('os').getenv('FEEDBACK_DB_PATH', str(DATA/'feedback.sqlite3')))

_SCHEMA = {
    # Only add columns that are safe to introduce with ALTER TABLE in SQLite.
    # Existing v7 databases may already have a different but compatible schema.
    'probability': 'REAL',
    'metadata': 'TEXT',
    'created_at': 'TEXT',
}

def init_db():
    DB.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(DB) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_url TEXT NOT NULL,
            label INTEGER NOT NULL CHECK(label IN (0,1)),
            source TEXT NOT NULL DEFAULT 'unknown',
            verified INTEGER NOT NULL DEFAULT 0,
            model_version TEXT,
            probability REAL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        cols={r[1] for r in con.execute('PRAGMA table_info(feedback)').fetchall()}
        # Forward migration for v7/v8 development databases created before
        # probability/metadata were present. Existing rows remain untouched.
        for name, ddl in _SCHEMA.items():
            if name not in cols:
                con.execute(f'ALTER TABLE feedback ADD COLUMN {name} {ddl}')
        # Preserve legacy metadata/risk values where the older schema used different names.
        cols={r[1] for r in con.execute('PRAGMA table_info(feedback)').fetchall()}
        if 'metadata' in cols and 'metadata_json' in cols:
            con.execute("UPDATE feedback SET metadata=COALESCE(metadata, metadata_json) WHERE metadata IS NULL")
        if 'probability' in cols and 'risk_score' in cols:
            con.execute("UPDATE feedback SET probability=COALESCE(probability, risk_score) WHERE probability IS NULL")
        con.execute('CREATE INDEX IF NOT EXISTS idx_feedback_url_verified ON feedback(normalized_url,verified,id DESC)')

def add(url,label,source,verified,model_version,probability,metadata):
    init_db(); normalized=normalize_url(url); meta=json.dumps(metadata or {},ensure_ascii=False)[:20000]
    # Support both the current schema and the legacy v7 database schema used by
    # existing local deployments.
    with sqlite3.connect(DB) as con:
        cols={r[1] for r in con.execute('PRAGMA table_info(feedback)').fetchall()}
        fields=['normalized_url','label','source','verified','model_version']
        values=[normalized,int(label),str(source)[:64],int(bool(verified)),model_version]
        if 'probability' in cols:
            fields.append('probability'); values.append(probability)
        elif 'risk_score' in cols:
            fields.append('risk_score'); values.append(probability)
        if 'metadata' in cols:
            fields.append('metadata'); values.append(meta)
        elif 'metadata_json' in cols:
            fields.append('metadata_json'); values.append(meta)
        if 'url_hash' in cols:
            fields.append('url_hash'); values.append(hashlib.sha256(normalized.encode('utf-8')).hexdigest())
        if 'received_at' in cols:
            fields.append('received_at'); values.append(time.time())
        if 'created_at' in cols and 'created_at' not in fields:
            fields.append('created_at'); values.append(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        marks=','.join('?' for _ in fields)
        con.execute(f"INSERT INTO feedback({','.join(fields)}) VALUES({marks})",values)

def verified_rows(limit=100000):
    init_db()
    with sqlite3.connect(DB) as con:
        return con.execute('SELECT normalized_url,label FROM feedback WHERE verified=1 ORDER BY id DESC LIMIT ?', (min(int(limit),100000),)).fetchall()
