import argparse
from database import init_db, fetch_all_reports
from preprocessing import enrich
from classifier import train_baseline
from config import MODEL_DIR, BERT_MODEL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", action="store_true", help="Also fine-tune DistilBERT")
    args = ap.parse_args()

    init_db()
    reports = fetch_all_reports()
    if not reports:
        print("No reports. Run ingestion or setup_demo first.")
        return

    texts, labels = [], []
    for r in reports:
        rec = enrich(dict(r))
        if rec["threat_type"] == "other":
            continue
        texts.append((r.get("title") or "") + ". " + (r.get("text") or ""))
        labels.append(rec["threat_type"])

    if len(set(labels)) < 2:
        print("Not enough label diversity to train.")
        return

    print(f"Training baseline on {len(texts)} samples; labels={sorted(set(labels))}")
    train_baseline(texts, labels)
    print("Saved baseline to", MODEL_DIR / "threat_clf.joblib")

    if args.bert:
        train_bert(texts, labels)

def train_bert(texts, labels):
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    label2id = {l: i for i, l in enumerate(sorted(set(labels)))}
    id2label = {i: l for l, i in label2id.items()}

    ds = Dataset.from_dict({
        "text": texts,
        "label": [label2id[l] for l in labels],
    })
    tok = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)

    def tok_fn(b):
        return tok(b["text"], truncation=True, padding="max_length", max_length=256)

    ds = ds.map(tok_fn, batched=True)
    ds = ds.train_test_split(test_size=0.15, seed=42)

    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "bert"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
    trainer.train()
    model.save_pretrained(MODEL_DIR / "bert")
    tok.save_pretrained(MODEL_DIR / "bert")
    print("Saved BERT to", MODEL_DIR / "bert")

if __name__ == "__main__":
    main()