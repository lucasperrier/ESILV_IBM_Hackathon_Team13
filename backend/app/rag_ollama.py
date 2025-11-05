import os
from typing import List, Dict

import requests
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def call_llama3_html_answer(question: str, contexts: List[Dict]) -> str:
    """
    Génère une réponse HTML à partir des FAQ contextuelles.

    contexts : liste de dicts {id, title, content}
    """
    context_blocks = []
    for c in contexts:
        context_blocks.append(
            f"[ID={c['id']}]\n"
            f"Titre : {c['title']}\n"
            f"Réponse : {c['content']}\n"
        )

    prompt = f"""
Tu es l'assistant officiel du helpcenter d'une école (PLV).
Tu dois répondre UNIQUEMENT en te basant sur les informations suivantes,
qui sont des fiches FAQ officielles.

Si l'information n'est pas présente ou est incomplète, tu le dis clairement
et tu proposes d'utiliser un formulaire ou un email de contact.

Réponds en HTML simple (balises <p>, <strong>, <ul>, <li>, <a>) adapté à un étudiant.
N'invente pas de nouvelles règles, n'hallucine pas.

Question de l'étudiant :
\"\"\"{question}\"\"\"

Connaissances (FAQ) :
{chr(10).join(context_blocks)}

Consignes :
- Réponds dans la langue de la question (ici en général français).
- Mets les informations importantes en <strong>.
- Si tu ne trouves pas l'information, écris un court paragraphe HTML qui dit
  que tu ne trouves pas d'information suffisante et propose de contacter
  la scolarité (mailto:scolarite@plv.example).

Réponse HTML :
""".strip()

    resp = requests.post(
        OLLAMA_URL.rstrip("/") + "/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Tu es un assistant de helpcenter universitaire."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

