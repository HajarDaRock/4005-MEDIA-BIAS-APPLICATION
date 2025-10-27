import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.textcnn import TextCNN


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


def build_vocab(texts: List[str], lowercase: bool, min_freq: int, pad_token: str = "<pad>", unk_token: str = "<unk>") -> Tuple[Dict[str, int], int, int]:
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
    return word2id, pad_id, unk_id


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


def split_train_val(texts: List[str], labels: List[int], val_frac: float, seed: int = 42):
    n = len(texts)
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
    parser.add_argument("--train_csv", required=True, help="Path to CSV with training data")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--device", default=None, help="cpu or cuda; default auto")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.train_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.text_col}' and '{args.label_col}'. Found: {df.columns.tolist()}")

    texts = df[args.text_col].astype(str).tolist()
    raw_labels = df[args.label_col].tolist()

    # Define label order
    id2label = ["Left", "Right", "Neutral"]
    labels, label2id = map_labels(raw_labels, id2label)

    # Split
    tr_texts, tr_labels, va_texts, va_labels = split_train_val(texts, labels, val_frac=args.val_frac, seed=args.seed)

    # Build vocab on training texts only
    lowercase = True
    word2id, pad_id, unk_id = build_vocab(tr_texts, lowercase=lowercase, min_freq=args.min_freq)

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
    }
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)

    print("Training complete. Artifacts saved to:")
    print(os.path.join(args.output_dir, "textcnn_state.pt"))
    print(os.path.join(args.output_dir, "vocab.json"))


if __name__ == "__main__":
    main()

