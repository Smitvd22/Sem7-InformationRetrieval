import os
import re
import numpy as np
from embeddings import encode

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [s.strip() for s in parts if len(s.strip()) > 30]

def answer_llm(query, retrieved_reports):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        context = "\n\n".join(
            f"[{r['id']}] {r['title']}\n{r.get('text','')[:800]}\nURL: {r.get('url','')}"
            for r in retrieved_reports
        )
        prompt = (
            "Answer the question using ONLY the context. "
            "Cite report IDs like [12]. If the context is insufficient, say so.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        cits = [
            f"[{r['id']}] {r.get('source')} — {r.get('title')} ({r.get('url')})"
            for r in retrieved_reports
        ]
        return {"answer": text, "citations": cits}
    except Exception as e:
        print("[chatbot] LLM error:", e)
        return None

def answer(query, retrieved_reports, top_k_sentences=5):
    if not retrieved_reports:
        return {"answer": "No relevant reports found.", "citations": []}

    llm = answer_llm(query, retrieved_reports)
    if llm:
        return llm

    corpus, meta = [], []
    for r in retrieved_reports:
        for s in split_sentences(r.get("text", "")):
            corpus.append(s)
            meta.append(r)
    if not corpus:
        for r in retrieved_reports:
            corpus.append(r.get("title", ""))
            meta.append(r)

    q = encode([query])[0]
    E = encode(corpus)
    sims = E @ q
    idxs = np.argsort(-sims)[:top_k_sentences]

    parts, citations = [], []
    for i in idxs:
        parts.append(corpus[i])
        r = meta[i]
        cit = f"[{r.get('id')}] {r.get('source','?')} — {r.get('title','?')} ({r.get('url','')})"
        if cit not in citations:
            citations.append(cit)
    return {"answer": " ".join(parts), "citations": citations}