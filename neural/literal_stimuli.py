"""Frozen literal input features for offline experiments; no language decoder."""
import json
from pathlib import Path

from safetensors import safe_open
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def literal_rows(texts):
    model_path = ROOT/'.runtime/minicpm-omni'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    index = json.loads((model_path/'model.safetensors.index.json').read_text())['weight_map']
    key = 'llm.model.embed_tokens.weight'
    with safe_open(model_path/index[key], framework='pt', device='cpu') as source:
        embedding = source.get_tensor(key)
        return [embedding[torch.tensor(tokenizer.encode(text, add_special_tokens=False))].half().clone()
                for text in texts]
