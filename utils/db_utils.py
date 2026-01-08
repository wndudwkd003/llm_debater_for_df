# /workspace/competition_xai/utils/db_utils.py

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modality TEXT NOT NULL,
            mode TEXT NOT NULL,          -- id / ood
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            path TEXT NOT NULL,

            y_true INTEGER,
            y_pred INTEGER,
            confidence REAL,
            correct INTEGER,

            saved_image TEXT,
            saved_gradcam TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(mode, dataset, split, sample_index, path)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,

            llm_model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,

            embedding BLOB,
            embedding_dim INTEGER,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(sample_id) REFERENCES evidence_samples(id)
        );
        """
    )
    conn.commit()


def _to_float32_blob(vec) -> Tuple[bytes, int]:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes(), int(arr.size)


def upsert_sample(
    conn: sqlite3.Connection,
    modality: str,
    mode: str,
    dataset: str,
    split: str,
    record: Dict[str, Any],
) -> int:
    """
    record: manifest/topk jsonl의 1행(dict)
    """
    conn.execute(
        """
        INSERT INTO evidence_samples (
            modality, mode, dataset, split, sample_index, path,
            y_true, y_pred, confidence, correct,
            saved_image, saved_gradcam
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mode, dataset, split, sample_index, path)
        DO UPDATE SET
            y_true=excluded.y_true,
            y_pred=excluded.y_pred,
            confidence=excluded.confidence,
            correct=excluded.correct,
            saved_image=excluded.saved_image,
            saved_gradcam=excluded.saved_gradcam
        ;
        """,
        (
            modality,
            mode,
            dataset,
            split,
            int(record["index"]),
            str(record["path"]),
            int(record.get("y_true", -1)),
            int(record.get("y_pred", -1)),
            float(record.get("confidence", 0.0)),
            1 if bool(record.get("correct", False)) else 0,
            str(record.get("saved_image", "")),
            str(record.get("saved_gradcam", "")),
        ),
    )
    conn.commit()

    cur = conn.execute(
        """
        SELECT id FROM evidence_samples
        WHERE mode=? AND dataset=? AND split=? AND sample_index=? AND path=?
        """,
        (mode, dataset, split, int(record["index"]), str(record["path"])),
    )
    row = cur.fetchone()
    return int(row[0])


def insert_explanation(
    conn: sqlite3.Connection,
    sample_id: int,
    llm_model: str,
    prompt: str,
    response: str,
    embedding: Optional[list] = None,
) -> int:
    emb_blob = None
    emb_dim = None
    if embedding is not None:
        emb_blob, emb_dim = _to_float32_blob(embedding)

    cur = conn.execute(
        """
        INSERT INTO llm_explanations (
            sample_id, llm_model, prompt, response, embedding, embedding_dim
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (int(sample_id), llm_model, prompt, response, emb_blob, emb_dim),
    )
    conn.commit()
    return int(cur.lastrowid)
