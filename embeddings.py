# embeddings.py

from transformers import AutoTokenizer, AutoModel
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model_and_tokenizer(model_name):
    """Model ve tokenizer yükler."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, ignore_mismatched_sizes=True)

    return tokenizer, model


def get_embeddings(texts, tokenizer, model):
    """Metinleri embedding vektörlerine dönüştürür."""
    inputs = tokenizer(texts, max_length=256, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Temsil için ortalama alınır

    return embeddings
