from embeddings import load_model_and_tokenizer, get_embeddings
import torch
import random
import csv
import os
import warnings
import time
import einops  # jinaai/jina-embeddings-v3 dil modeli için gerekli

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_embeddings(embeddings, filename):
    folder_name = os.path.dirname(filename)
    os.makedirs(folder_name, exist_ok=True)
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


def process_model_embeddings(model_name, inputs, outputs, folder_name, batch_size=10):
    """Verilen model için embedding hesaplayıp dosyaya kaydeder."""
    # Model ve tokenizer’ı yükle
    tokenizer, model = load_model_and_tokenizer(model_name)
    input_embeddings = []
    output_embeddings = []

    # Dosya ismindeki özel karakterleri değiştir
    safe_model_name = model_name.replace("/", "_")

    # Batch olarak embedding hesapla
    for i in range(0, len(inputs), batch_size):
        batch_questions = inputs[i:i + batch_size]
        batch_answers = outputs[i:i + batch_size]

        input_embeddings.append(get_embeddings(batch_questions, tokenizer, model))
        output_embeddings.append(get_embeddings(batch_answers, tokenizer, model))

    # Embedding'leri dosyaya kaydet
    save_embeddings(torch.cat(input_embeddings), f'{folder_name}/{safe_model_name}_input_embeddings.pt')
    save_embeddings(torch.cat(output_embeddings), f'{folder_name}/{safe_model_name}_output_embeddings.pt')


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
        "intfloat/multilingual-e5-large",
        "Alibaba-NLP/gte-multilingual-base",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "jinaai/jina-embeddings-v3",
        "izhx/udever-bloom-560m"
    ]

    folder_q2a = "embeddings_q2a"
    folder_a2q = "embeddings_a2q"
    # Her model için embedding işlemi
    for model_name in model_names:
        print(f"Processing embeddings for model: {model_name}")
        process_model_embeddings(model_name, selected_questions, selected_answers, folder_q2a)
        process_model_embeddings(model_name, selected_answers, selected_questions, folder_a2q)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
