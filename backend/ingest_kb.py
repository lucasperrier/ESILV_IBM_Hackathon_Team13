import os
import pickle
from pathlib import Path

import faiss
import pandas as pd
from ftfy import fix_text
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_URI = os.getenv("DB_URI")
CSV_PATH = os.getenv("CSV_PATH", "Questions-Export-2025-October-27-1237.csv")
CSV_SEP = os.getenv("CSV_SEP", ";")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "data/id_map.pkl")

# Mapping colonnes CSV -> schéma faqs
COLUMN_MAP = {
    "id": "id",
    "Title": "title",
    "Content": "content",
    "Date": "date",
    "Post Type": "post_type",

    "Langues": "languages",
    "Langue": "languages",

    "Thématiques": "topics",
    "Th�matiques": "topics",

    "Utilisateurs": "users",

    "Écoles": "schools",
    "Ecoles": "schools",
    "�coles": "schools",

    "Status": "status",
}


def clean_text(s) -> str:
    """Nettoyage simple + correction d'encodage."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = fix_text(s)  # répare Fran�ais -> Français
    return s.strip()


def main():
    if not DB_URI:
        raise RuntimeError("DB_URI n'est pas défini dans l'environnement (.env).")

    engine = create_engine(DB_URI, pool_pre_ping=True)

    print(f"[INGEST] Lecture du CSV : {CSV_PATH}")
    try:
        df = pd.read_csv(
            CSV_PATH,
            sep=CSV_SEP,
            encoding="utf-8",
            engine="python",
        )
    except UnicodeDecodeError:
        print("[WARN] UTF-8 decoding failed — retrying with latin1 encoding...")
        df = pd.read_csv(
            CSV_PATH,
            sep=CSV_SEP,
            encoding="latin1",  # compatible accents Windows
            engine="python",
        )


    # Normalisation des noms de colonnes
    renamed = {}
    for col in df.columns:
        fixed = fix_text(str(col))
        if fixed in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[fixed]
        elif col in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[col]
        else:
            renamed[col] = fixed.lower()

    df = df.rename(columns=renamed)

    # Colonnes attendues
    expected = [
        "id",
        "title",
        "content",
        "date",
        "post_type",
        "languages",
        "topics",
        "users",
        "schools",
        "status",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    df = df[expected]

    # Nettoyage texte
    for col in [
        "title",
        "content",
        "languages",
        "topics",
        "users",
        "schools",
        "post_type",
        "status",
    ]:
        df[col] = df[col].apply(clean_text)

    # Forcer id en int si possible
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # Optionnel : on garde seulement les entrées 'publish'
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].str.lower() == "publish"]
        print(f"[INGEST] Filtrage status=publish : {before} → {len(df)} lignes")

    print("[INGEST] Insertion / mise à jour des FAQ dans MariaDB...")
    with engine.begin() as conn:
        for _, row in df.iterrows():
            payload = row.to_dict()
            conn.execute(
                text(
                    """
                    INSERT INTO faqs (
                        id, title, content, date, post_type, languages,
                        topics, users, schools, status
                    ) VALUES (
                        :id, :title, :content, :date, :post_type, :languages,
                        :topics, :users, :schools, :status
                    )
                    ON DUPLICATE KEY UPDATE
                        title = VALUES(title),
                        content = VALUES(content),
                        date = VALUES(date),
                        post_type = VALUES(post_type),
                        languages = VALUES(languages),
                        topics = VALUES(topics),
                        users = VALUES(users),
                        schools = VALUES(schools),
                        status = VALUES(status)
                    """
                ),
                payload,
            )

    print("[INGEST] Génération des embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = (df["title"].fillna("") + "\n" + df["content"].fillna("")).tolist()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = emb.astype("float32")

    # Création du dossier pour FAISS si besoin
    faiss_path = Path(FAISS_INDEX_PATH)
    idmap_path = Path(ID_MAP_PATH)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    idmap_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INGEST] Construction de l'index FAISS...")
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product sur vecteurs normalisés
    index.add(emb)
    faiss.write_index(index, str(faiss_path))

    id_map = df["id"].astype(int).tolist()
    with open(idmap_path, "wb") as f:
        pickle.dump(id_map, f)

    print(f"[INGEST] Terminé. {len(df)} FAQ indexées.")
    print(f"[INGEST] Index FAISS : {faiss_path}")
    print(f"[INGEST] ID map      : {idmap_path}")


if __name__ == "__main__":
    main()

