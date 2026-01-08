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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    # row: (cid, name, type, notnull, dflt_value, pk)
    return {str(r[1]) for r in cur.fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, col_defs: Dict[str, str]):
    """
    기존 DB가 예전 스키마로 만들어져 있어도, 필요한 컬럼을 ALTER TABLE로 추가합니다.
    col_defs: {"col_name": "SQL_TYPE", ...}
    """
    existing = _table_columns(conn, table)
    for name, sql_type in col_defs.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type};")
    conn.commit()


def _init_schema(conn: sqlite3.Connection):
    # 새 DB 생성 시 최신 스키마
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

            -- 이미지 임베딩 저장 (추가)
            image_embedding BLOB,
            image_embedding_dim INTEGER,
            image_embedding_model TEXT,

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

    # 예전 DB(컬럼 없는 상태) 마이그레이션
    _ensure_columns(
        conn,
        "evidence_samples",
        {
            "image_embedding": "BLOB",
            "image_embedding_dim": "INTEGER",
            "image_embedding_model": "TEXT",
        },
    )


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
    image_embedding: Optional[list] = None,
    image_emb_dim: Optional[int] = None,
    image_emb_model: Optional[str] = None,
) -> int:
    """
    record: manifest/topk jsonl의 1행(dict)
    image_embedding: CLIP 이미지 임베딩 벡터(list[float])
    """
    emb_blob = None
    emb_dim_final = None

    if image_embedding is not None:
        emb_blob, emb_dim_final = _to_float32_blob(image_embedding)
        if image_emb_dim is not None and int(image_emb_dim) != int(emb_dim_final):
            raise ValueError(
                f"image_emb_dim mismatch: got {image_emb_dim}, actual {emb_dim_final}"
            )
    elif image_emb_dim is not None:
        # 벡터 없이 dim만 오는 경우는 비정상: 저장할 게 없으니 무시하지 말고 에러로 막습니다.
        raise ValueError("image_emb_dim was provided but image_embedding is None")

    conn.execute(
        """
        INSERT INTO evidence_samples (
            modality, mode, dataset, split, sample_index, path,
            y_true, y_pred, confidence, correct,
            saved_image, saved_gradcam,
            image_embedding, image_embedding_dim, image_embedding_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mode, dataset, split, sample_index, path)
        DO UPDATE SET
            y_true=excluded.y_true,
            y_pred=excluded.y_pred,
            confidence=excluded.confidence,
            correct=excluded.correct,
            saved_image=excluded.saved_image,
            saved_gradcam=excluded.saved_gradcam,
            image_embedding=excluded.image_embedding,
            image_embedding_dim=excluded.image_embedding_dim,
            image_embedding_model=excluded.image_embedding_model
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
            emb_blob,
            int(emb_dim_final) if emb_dim_final is not None else None,
            str(image_emb_model) if image_emb_model is not None else None,
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
