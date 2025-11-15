import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.textcnn import TextCNN

# Evaluation/plots
try:
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
except Exception:
    classification_report = None  # type: ignore
    plt = None  # type: ignore


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if not isinstance(text, str):
        text = ""
    if lowercase:
        text = text.lower()
    return re.findall(r"\b\w+\b", text)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int, lowercase: bool, unk_id: int, pad_id: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.lowercase = lowercase
        self.unk_id = unk_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.texts)

    def _numericalize(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(tok, self.unk_id) for tok in tokens]

    def _pad_or_truncate(self, ids: List[int]) -> List[int]:
        if len(ids) >= self.max_len:
            return ids[: self.max_len]
        return ids + [self.pad_id] * (self.max_len - len(ids))

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = simple_tokenize(text, lowercase=self.lowercase)
        ids = self._numericalize(tokens)
        ids = self._pad_or_truncate(ids)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_vocab(texts: List[str], lowercase: bool, min_freq: int, pad_token: str = "<pad>", unk_token: str = "<unk>") -> Tuple[Dict[str, int], int, int, Counter]:
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(t, lowercase=lowercase)
        counter.update(tokens)
    # Reserve 0: PAD, 1: UNK
    word2id: Dict[str, int] = {pad_token: 0, unk_token: 1}
    for word, freq in counter.items():
        if freq >= min_freq and word not in word2id:
            word2id[word] = len(word2id)
    pad_id = word2id[pad_token]
    unk_id = word2id[unk_token]
    return word2id, pad_id, unk_id, counter


def map_labels(raw_labels: List, id2label: List[str]) -> Tuple[List[int], Dict[str, int]]:
    # If labels are strings like Left/Right/Neutral, map using id2label order
    label2id = {lab: i for i, lab in enumerate(id2label)}
    mapped = []
    for y in raw_labels:
        if isinstance(y, str):
            y_clean = y.strip()
            # Robust mapping allowing different casings
            for lab in id2label:
                if y_clean.lower() == lab.lower():
                    mapped.append(label2id[lab])
                    break
            else:
                raise ValueError(f"Unknown label '{y}'. Expected one of {id2label} or numeric 0..{len(id2label)-1}.")
        else:
            # Assume numeric
            yi = int(y)
            if 0 <= yi < len(id2label):
                mapped.append(yi)
            else:
                raise ValueError(f"Numeric label {yi} out of range for classes {len(id2label)}")
    return mapped, label2id


def split_train_val(texts: List[str], labels: List[int], val_frac: float, seed: int = 42, stratify: bool = True):
    n = len(texts)
    if not stratify:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(n * val_frac))
        val_idx = set(idx[:n_val])
        tr_texts, tr_labels, va_texts, va_labels = [], [], [], []
        for i in range(n):
            if i in val_idx:
                va_texts.append(texts[i])
                va_labels.append(labels[i])
            else:
                tr_texts.append(texts[i])
                tr_labels.append(labels[i])
        return tr_texts, tr_labels, va_texts, va_labels

    # Stratified split: keep label proportions in train/val
    label_to_indices: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        label_to_indices.setdefault(y, []).append(i)
    rnd = random.Random(seed)
    tr_idx: List[int] = []
    va_idx: List[int] = []
    for y, inds in label_to_indices.items():
        inds = inds[:]
        rnd.shuffle(inds)
        n_val = max(1, int(len(inds) * val_frac)) if len(inds) > 1 else 1 if val_frac > 0 else 0
        va_idx.extend(inds[:n_val])
        tr_idx.extend(inds[n_val:])
    tr_texts = [texts[i] for i in tr_idx]
    tr_labels = [labels[i] for i in tr_idx]
    va_texts = [texts[i] for i in va_idx]
    va_labels = [labels[i] for i in va_idx]
    return tr_texts, tr_labels, va_texts, va_labels


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
    return total_loss / max(1, total), correct / max(1, total)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
    return total_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train a TextCNN classifier for political bias")
    parser.add_argument("--train_csv", default="data/train.csv", help="Path to CSV with training data (default: data/train.csv)")
    parser.add_argument("--text_col", default="text", help="Column name containing article text")
    parser.add_argument("--label_col", default="label", help="Column name containing labels")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--filter_sizes", type=str, default="3,4,5")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--stratify", type=lambda s: str(s).lower() not in {"0","false","no","n"}, default=True,
                        help="Stratify train/val split by label (default: True)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--device", default=None, help="cpu or cuda; default auto")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure training CSV exists or guide the user clearly
    if not os.path.exists(args.train_csv):
        fallback = os.path.join("data", "sample_train.csv")
        if os.path.exists(fallback):
            print(
                f"Training CSV not found: {args.train_csv}\n"
                f"Falling back to sample dataset: {fallback}\n"
                "To use your own data, pass --train_csv or create data/train.csv with prepare_kaggle_data.py."
            )
            args.train_csv = fallback
        else:
            raise SystemExit(
                (
                    f"Training CSV not found: {args.train_csv}\n"
                    "Provide a valid path via --train_csv, or create one with:\n"
                    "  python prepare_kaggle_data.py --inputs \"data\\*.csv\" --output data/train.csv\n"
                    "Then run:\n"
                    "  python train_textcnn.py --train_csv data/train.csv\n"
                )
            )

    df = pd.read_csv(args.train_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.text_col}' and '{args.label_col}'. Found: {df.columns.tolist()}")

    texts = df[args.text_col].astype(str).tolist()
    raw_labels = df[args.label_col].tolist()

    # Define label order
    id2label = ["Left", "Right", "Neutral"]
    labels, label2id = map_labels(raw_labels, id2label)

    # Split
    tr_texts, tr_labels, va_texts, va_labels = split_train_val(texts, labels, val_frac=args.val_frac, seed=args.seed, stratify=args.stratify)

    # Build vocab on training texts only
    lowercase = True
    word2id, pad_id, unk_id, freq_counter = build_vocab(tr_texts, lowercase=lowercase, min_freq=args.min_freq)

    # Datasets
    train_ds = TextDataset(tr_texts, tr_labels, word2id, args.max_len, lowercase, unk_id, pad_id)
    val_ds = TextDataset(va_texts, va_labels, word2id, args.max_len, lowercase, unk_id, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    filter_sizes = [int(x) for x in args.filter_sizes.split(",") if x.strip()]
    model = TextCNN(
        vocab_size=len(word2id),
        embed_dim=args.embedding_dim,
        num_classes=len(id2label),
        filter_sizes=filter_sizes,
        num_filters=args.num_filters,
        dropout=args.dropout,
        padding_idx=pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
        # Save best by val loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "textcnn_state.pt"))

    # Save config/vocab
    # Build a readable summary for humans
    top_tokens = [
        {"token": tok, "count": int(cnt)}
        for tok, cnt in freq_counter.most_common(50)
        if tok in word2id
    ]

    config = {
        "word2id": word2id,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "max_len": args.max_len,
        "lowercase": True,
        "id2label": id2label,
        "embedding_dim": args.embedding_dim,
        "filter_sizes": filter_sizes,
        "num_filters": args.num_filters,
        "dropout": args.dropout,
        "token_count": len(word2id),
        "top_tokens": top_tokens,
        "notes": "word2id maps tokens to integer IDs used by the model. Tokens are derived from training text; common news terms like 'lawmakers' appear because they are frequent in the corpus, not because of bias by themselves."
    }
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("Training complete. Artifacts saved to:")
    print(os.path.join(args.output_dir, "textcnn_state.pt"))
    print(os.path.join(args.output_dir, "vocab.json"))

    # Evaluate on validation and save metrics + visualization
    try:
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu().tolist()
                y_pred.extend(pred)
                y_true.extend(y.cpu().tolist())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = None
        if classification_report is not None:
            # Force all classes to appear; avoid empty plots when a class is absent
            label_ids = list(range(len(id2label)))
            report = classification_report(
                y_true,
                y_pred,
                labels=label_ids,
                target_names=id2label,
                zero_division=0,
                output_dict=True,
            )
            # Save JSON report
            rep_path = os.path.join(metrics_dir, f"report_{ts}.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": ts,
                    "val_loss": best_val_loss,
                    "val_acc": va_acc,
                    "config": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "max_len": args.max_len,
                        "min_freq": args.min_freq,
                        "lr": args.lr,
                        "embedding_dim": args.embedding_dim,
                        "num_filters": args.num_filters,
                        "filter_sizes": filter_sizes,
                        "dropout": args.dropout,
                        "stratify": args.stratify,
                        "val_frac": args.val_frac,
                    },
                    "report": report
                }, f, ensure_ascii=False, indent=2)

            # Append to CSV history (macro/weighted F1)
            macro_f1 = report.get("macro avg", {}).get("f1-score", None)
            weighted_f1 = report.get("weighted avg", {}).get("f1-score", None)
            hist_path = os.path.join(metrics_dir, "history.csv")
            header_needed = not os.path.exists(hist_path)
            import csv
            with open(hist_path, "a", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                if header_needed:
                    writer.writerow(["timestamp","val_loss","val_acc","macro_f1","weighted_f1","epochs","batch","max_len","min_freq","lr","embed","filters","filter_sizes","dropout","stratify","val_frac"])
                writer.writerow([ts, best_val_loss, va_acc, macro_f1, weighted_f1, args.epochs, args.batch_size, args.max_len, args.min_freq, args.lr, args.embedding_dim, args.num_filters, ";".join(map(str, filter_sizes)), args.dropout, args.stratify, args.val_frac])

            # Visualization: per-class Precision/Recall/F1 bar chart
            if plt is not None:
                classes = id2label
                # Ensure a value for every class, even if 0
                precisions = [float(report.get(c, {}).get("precision", 0.0)) for c in classes]
                recalls = [float(report.get(c, {}).get("recall", 0.0)) for c in classes]
                f1s = [float(report.get(c, {}).get("f1-score", 0.0)) for c in classes]
                idx = list(range(len(classes)))
                w = 0.25
                x_p = [i - w for i in idx]
                x_r = idx
                x_f = [i + w for i in idx]
                plt.figure(figsize=(8, 4))
                plt.bar(x_p, precisions, width=w, label="Precision")
                plt.bar(x_r, recalls, width=w, label="Recall")
                plt.bar(x_f, f1s, width=w, label="F1-Score")
                plt.xticks(idx, classes)
                plt.ylim(0, 1.0)
                plt.title("Validation Metrics by Class")
                plt.ylabel("Score")
                plt.legend()
                png_path = os.path.join(metrics_dir, f"report_{ts}.png")
                plt.tight_layout()
                plt.savefig(png_path)
                try:
                    plt.close()
                except Exception:
                    pass
                print(f"Saved metrics: {rep_path} and {png_path}")
        else:
            print("sklearn not installed; skipping detailed metrics. Install scikit-learn to enable.")
    except Exception as e:
        print(f"[EVAL WARN] Failed to generate metrics: {e}")


if __name__ == "__main__":
    main()

