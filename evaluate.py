import torch
from similarity import cosine_similarity
import matplotlib.pyplot as plt
import os
import time


def load_embeddings(filename):
    return torch.load(filename)


def calculate_top_k_accuracy(sorular_embeddings, cevaplar_embeddings, k=5):
    """Top-k başarı oranlarını hesaplar."""
    accuracies = [0] * k
    num_samples = sorular_embeddings.shape[0]

    for idx in range(num_samples):
        soru_embedding = sorular_embeddings[idx].unsqueeze(0)
        similarities = cosine_similarity(soru_embedding, cevaplar_embeddings).tolist()
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]

        # Gerçek cevap indeksini kontrol et
        for j in range(k):
            if idx in sorted_indices[:j + 1]:
                accuracies[j] += 1

    return [accuracy / num_samples for accuracy in accuracies]


def plot_accuracies(model_name, q_to_a_accuracies, a_to_q_accuracies):
    """Her model için tek bir grafik dosyasında soru->cevap ve cevap->soru doğruluklarını çiz ve kaydet."""
    top_k_labels = [f"Top-{i + 1}" for i in range(len(q_to_a_accuracies))]

    plt.figure(figsize=(10, 6))
    plt.plot(top_k_labels, q_to_a_accuracies, marker='o', linestyle='-', color='b', label="Question->Answer")
    plt.plot(top_k_labels, a_to_q_accuracies, marker='o', linestyle='-', color='r', label="Answer->Question")
    plt.title(f"{model_name} - Top-K Accuracy")
    plt.xlabel("Top-K")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Grafiklerin kaydedileceği dizini oluştur
    output_dir = "combined_accuracy_plots"
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    plt.savefig(f"{output_dir}/{safe_model_name}_combined_accuracy.png")
    plt.close()


def main():
    model_names = [
        "dbmdz/bert-base-turkish-cased",
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-large-en-v1.5",
        "Alibaba-NLP/gte-base-en-v1.5",
        "intfloat/multilingual-e5-large-instruct",
        "Alibaba-NLP/gte-multilingual-base"
    ]
    folder_q2a = "embeddings_q2a"
    folder_a2q = "embeddings_a2q"

    for model_name in model_names:
        safe_model_name = model_name.replace("/", "_")

        # Soru->Cevap doğrulukları
        # print(f"Calculating Q->A accuracy for model: {model_name}")
        sorular_embeddings = load_embeddings(f'{folder_q2a}/{safe_model_name}_question_embeddings.pt')
        cevaplar_embeddings = load_embeddings(f'{folder_q2a}/{safe_model_name}_answer_embeddings.pt')
        q_to_a_accuracies = calculate_top_k_accuracy(sorular_embeddings, cevaplar_embeddings)

        # Top-1 ve Top-5 değerlerini konsola yazdır
        # print(f"{model_name} - Q->A Top-1 Accuracy: {q_to_a_accuracies[0]:.2f}")
        # print(f"{model_name} - Q->A Top-5 Accuracy: {q_to_a_accuracies[4]:.2f}")

        # Cevap->Soru doğrulukları
        # print(f"Calculating A->Q accuracy for model: {model_name}")
        sorular_embeddings = load_embeddings(f'{folder_a2q}/{safe_model_name}_question_embeddings.pt')
        cevaplar_embeddings = load_embeddings(f'{folder_a2q}/{safe_model_name}_answer_embeddings.pt')
        a_to_q_accuracies = calculate_top_k_accuracy(sorular_embeddings, cevaplar_embeddings)

        # Top-1 ve Top-5 değerlerini konsola yazdır
        # print(f"{model_name} - A->Q Top-1 Accuracy: {a_to_q_accuracies[0]:.2f}")
        # print(f"{model_name} - A->Q Top-5 Accuracy: {a_to_q_accuracies[4]:.2f}")

        print(
            f"{model_name} - Q->A Top-1 Accuracy: {q_to_a_accuracies[0]:.2f} Top-5 Accuracy: {q_to_a_accuracies[4]:.2f}")
        print(
            f"{model_name} - A->Q Top-1 Accuracy: {a_to_q_accuracies[0]:.2f} Top-5 Accuracy: {a_to_q_accuracies[4]:.2f}")

        # Grafikleştir ve kaydet
        plot_accuracies(model_name, q_to_a_accuracies, a_to_q_accuracies)


if __name__ == '__main__':
    # Başlangıç zamanı
    start_time = time.time()
    main()
    # Bitiş zamanı
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
