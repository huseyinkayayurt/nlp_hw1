# similarity.py

import torch

def cosine_similarity(embedding1, embedding2):
    """İki embedding vektörü arasında kosinüs benzerliği hesaplar."""
    cos = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cos
