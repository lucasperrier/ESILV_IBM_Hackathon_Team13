import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import bleach
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langdetect import detect
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from .rag_ollama import call_llama3_html_answer

# ----------------- Chargement de la config -----------------

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DB_URI = os.getenv("DB_URI")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "data/id_map.pkl")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))

if not DB_URI:
    raise RuntimeError("DB_URI n'est pas défini dans l'environnement (.env).")

# ----------------- Initialisation des composants -----------------

app = FastAPI(title="PLV Helpcenter RAG (Llama3 + FAISS)")

engine = create_engine(DB_URI, pool_pre_ping=True)
model = SentenceTransformer(EMBEDDING_MODEL)

# FAISS index + mapping ids
if not Path(FAISS_INDEX_PATH).exists() or not Path(ID_MAP_PATH).exists():
    raise RuntimeError(
        "Index FAISS ou ID map manquants. Lance d'abord `python ingest_kb.py`."
    )

index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "rb") as f:
    id_map: List[int] = pickle.load(f)

# Sanitization HTML
ALLOWED_TAGS = ["p", "br", "strong", "em", "ul", "ol", "li", "a"]
ALLOWED_ATTRS = {"a": ["href", "title", "target", "rel"]}


def sanitize_html(html: str) -> str:
    return bleach.clean(html or "", tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)


# ----------------- Modèles Pydantic -----------------

class AskRequest(BaseModel):
    query: str
    lang: Optional[str] = None
    user_id: Optional[str] = None


class AskResponse(BaseModel):
    interaction_id: Optional[int]
    decision: str
    html: str


# ----------------- Endpoints -----------------

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    # Détection langue (best effort)
    try:
        lang = payload.lang or detect(query)
    except Exception:
        lang = payload.lang or "fr"

    # 1) Embedding + recherche FAISS
    qvec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qvec, RETRIEVAL_TOP_K)
    scores = D[0]
    idxs = I[0]

    # Si aucun résultat ou score trop bas => redirect direct
    if len(idxs) == 0 or float(scores[0]) < SCORE_THRESHOLD:
        html = (
            "<p>Je n’ai pas trouvé d’information suffisamment pertinente dans la base de connaissances.</p>"
            "<p>Vous pouvez contacter la scolarité : "
            "<a href='mailto:scolarite@plv.example'>scolarite@plv.example</a>.</p>"
        )
        html = sanitize_html(html)
        interaction_id = log_interaction(
            user_id=payload.user_id,
            query=query,
            detected_lang=lang,
            retrieved_ids=[],
            chosen_faq_id=None,
            decision="redirect",
            answer_html=html,
        )
        return AskResponse(
            interaction_id=interaction_id,
            decision="redirect",
            html=html,
        )

    faq_ids = [id_map[i] for i in idxs]
    retrieved_ids_str = ",".join(str(i) for i in faq_ids)

    # 2) Récupération des FAQ dans MariaDB
    # on garde le même ordre que faq_ids
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, title, content FROM faqs WHERE id IN :ids"),
            {"ids": tuple(faq_ids)},
        ).mappings().all()

    row_by_id = {int(r["id"]): r for r in rows}
    contexts = []
    for faq_id in faq_ids:
        r = row_by_id.get(int(faq_id))
        if not r:
            continue
        contexts.append(
            {
                "id": int(r["id"]),
                "title": (r["title"] or "")[:150],
                "content": (r["content"] or "")[:1000],
            }
        )

    if not contexts:
        html = (
            "<p>Aucune fiche FAQ correspondante n’a été trouvée.</p>"
            "<p>Vous pouvez contacter la scolarité : "
            "<a href='mailto:scolarite@plv.example'>scolarite@plv.example</a>.</p>"
        )
        html = sanitize_html(html)
        interaction_id = log_interaction(
            user_id=payload.user_id,
            query=query,
            detected_lang=lang,
            retrieved_ids=[],
            chosen_faq_id=None,
            decision="redirect",
            answer_html=html,
        )
        return AskResponse(
            interaction_id=interaction_id,
            decision="redirect",
            html=html,
        )

    # 3) RAG : génération HTML via Llama3 + contextes
    raw_html = call_llama3_html_answer(query, contexts)
    html = sanitize_html(raw_html)

    # On prend le premier contexte comme "chosen_faq_id" principal
    chosen_faq_id = contexts[0]["id"]

    interaction_id = log_interaction(
        user_id=payload.user_id,
        query=query,
        detected_lang=lang,
        retrieved_ids=faq_ids,
        chosen_faq_id=chosen_faq_id,
        decision="answer",
        answer_html=html,
    )

    return AskResponse(
        interaction_id=interaction_id,
        decision="answer",
        html=html,
    )


# ----------------- Logging interactions -----------------

def log_interaction(
    user_id: Optional[str],
    query: str,
    detected_lang: str,
    retrieved_ids: List[int],
    chosen_faq_id: Optional[int],
    decision: str,
    answer_html: str,
) -> Optional[int]:
    retrieved_str = ",".join(str(i) for i in retrieved_ids) if retrieved_ids else None

    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO interactions (
                    user_id, query_text, detected_lang,
                    retrieved_faq_ids, chosen_faq_id,
                    decision, answer_html
                ) VALUES (
                    :user_id, :query_text, :detected_lang,
                    :retrieved_faq_ids, :chosen_faq_id,
                    :decision, :answer_html
                )
                """
            ),
            {
                "user_id": user_id,
                "query_text": query,
                "detected_lang": detected_lang,
                "retrieved_faq_ids": retrieved_str,
                "chosen_faq_id": chosen_faq_id,
                "decision": decision,
                "answer_html": answer_html,
            },
        )
        return res.lastrowid

