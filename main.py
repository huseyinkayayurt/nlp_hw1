from datasets import load_dataset
from embeddings import load_model_and_tokenizer, get_embeddings
from evaluate import calculate_top_k_accuracy
import torch
import random
import csv
import os
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_embeddings(embeddings, filename):
    torch.save(embeddings, filename)


def load_data_from_csv(file_path):
    questions = []
    answers = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            questions.append(row['sorular'])
            answers.append(row['çıktı'])

    return questions, answers


def process_model_embeddings(model_name, questions, answers, folder_name, batch_size=10):
    """Verilen model için embedding hesaplayıp dosyaya kaydeder."""
    # Model ve tokenizer’ı yükle
    tokenizer, model = load_model_and_tokenizer(model_name)
    question_embeddings = []
    answer_embeddings = []

    # Dosya ismindeki özel karakterleri değiştir
    safe_model_name = model_name.replace("/", "_")

    # Batch olarak embedding hesapla
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]

        question_embeddings.append(get_embeddings(batch_questions, tokenizer, model))
        answer_embeddings.append(get_embeddings(batch_answers, tokenizer, model))

    # Embedding'leri dosyaya kaydet
    save_embeddings(torch.cat(question_embeddings), f'{folder_name}/{safe_model_name}_question_embeddings.pt')
    save_embeddings(torch.cat(answer_embeddings), f'{folder_name}/{safe_model_name}_answer_embeddings.pt')


def main():
    """Ana fonksiyon."""
    file_path = 'data_set.csv'  # CSV dosyanızın yolu
    questions, answers = load_data_from_csv(file_path)

    # Rasgele 1000 soru-cevap seçimi
    indices = random.sample(range(len(questions)), 1000)
    selected_questions = [questions[i] for i in indices]
    selected_answers = [answers[i] for i in indices]

    # Kullanılacak modeller
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
    # Her model için embedding işlemi
    for model_name in model_names:
        print(f"Processing embeddings for model: {model_name}")
        process_model_embeddings(model_name, selected_questions, selected_answers, folder_q2a)

    for model_name in model_names:
        print(f"Processing embeddings for model: {model_name}")
        process_model_embeddings(model_name, selected_answers, selected_questions, folder_a2q)


if __name__ == '__main__':
    # Başlangıç zamanı
    start_time = time.time()
    main()
    # Bitiş zamanı
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Zamanı saat, dakika ve saniye olarak formatla
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
