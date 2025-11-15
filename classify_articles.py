import json
import os
import re
from typing import Dict, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _torch_err:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

# Local TextCNN definition will be in models/textcnn.py
try:
    from models.textcnn import TextCNN
except Exception as e:
    TextCNN = None  # Will be checked at runtime


_MODEL = None
_DEVICE = None if torch is None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CONFIG: Dict = None
_VOCAB: Dict[str, int] = None


def _simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if not text:
        return []
    if lowercase:
        text = text.lower()
    # basic word tokenizer: alphanumeric word boundaries
    return re.findall(r"\b\w+\b", text)


def _load_artifacts():
    global _MODEL, _CONFIG, _VOCAB
    if _MODEL is not None and _CONFIG is not None and _VOCAB is not None:
        return

    config_path = os.path.join("models", "vocab.json")
    weights_path = os.path.join("models", "textcnn_state.pt")

    if TextCNN is None or torch is None:
        print("[MODEL ERROR] TextCNN module not available. Ensure models/textcnn.py exists.")
        return

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print("[MODEL INFO] Missing artifacts. Train the model to create models/vocab.json and models/textcnn_state.pt")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
        _VOCAB = _CONFIG.get("word2id", {})
        num_classes = len(_CONFIG.get("id2label", ["Left", "Right", "Neutral"]))
        model = TextCNN(
            vocab_size=len(_VOCAB),
            embed_dim=int(_CONFIG.get("embedding_dim", 100)),
            num_classes=num_classes,
            filter_sizes=_CONFIG.get("filter_sizes", [3, 4, 5]),
            num_filters=int(_CONFIG.get("num_filters", 100)),
            dropout=float(_CONFIG.get("dropout", 0.5)),
            padding_idx=int(_CONFIG.get("pad_id", 0)),
        )
        state = torch.load(weights_path, map_location=_DEVICE)
        model.load_state_dict(state)
        model.to(_DEVICE)
        model.eval()
        _MODEL = model
        print("[MODEL] TextCNN loaded successfully.")
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load TextCNN artifacts: {e}")
        _MODEL = None
        _CONFIG = None
        _VOCAB = None


def _numericalize(tokens: List[str], vocab: Dict[str, int], unk_id: int) -> List[int]:
    return [vocab.get(tok, unk_id) for tok in tokens]


def _pad_or_truncate(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def classify_bias(article_text: str) -> str:
    try:
        if _MODEL is None:
            _load_artifacts()

        if _MODEL is None or _CONFIG is None or _VOCAB is None:
            return "Model not available. Please train using train_textcnn.py."

        max_len = int(_CONFIG.get("max_len", 1000))
        lowercase = bool(_CONFIG.get("lowercase", True))
        pad_id = int(_CONFIG.get("pad_id", 0))
        unk_id = int(_CONFIG.get("unk_id", 1))
        id2label = _CONFIG.get("id2label", ["Left", "Right", "Neutral"])

        # Shorten input akin to previous behavior but keep consistent with model's max_len
        tokens = _simple_tokenize(article_text or "", lowercase=lowercase)
        ids = _numericalize(tokens, _VOCAB, unk_id)
        ids = _pad_or_truncate(ids, max_len=max_len, pad_id=pad_id)

        if torch is None:
            return "Model not available. Please install torch and train using train_textcnn.py."
        x = torch.tensor([ids], dtype=torch.long, device=_DEVICE)
        with torch.no_grad():
            logits = _MODEL(x)
            pred = int(torch.argmax(logits, dim=1).item())
        # Safety
        if 0 <= pred < len(id2label):
            return id2label[pred]
        return "Neutral"
    except Exception as e:
        print(f"[MODEL ERROR] Exception occurred: {e}")
        return "Back-end error. Please try again."
