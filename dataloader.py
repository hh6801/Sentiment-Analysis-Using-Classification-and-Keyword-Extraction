import json

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def load_data(file_path, train_ratio=0.8):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            data.append(json.loads(line.strip()))
            if i == 50000:  # Giới hạn số dòng (ví dụ 500 mẫu)
                break

    # Chuyển đổi thành DataFrame
    df = pd.DataFrame(data)

    # Chuẩn hóa nhãn cảm xúc
    def map_stars_to_sentiment(stars):
        if stars <= 2.0:
            return 0  # Negative
        elif stars == 3.0:
            return 1  # Neutral
        else:
            return 2  # Positive

    df['sentiment'] = df['stars'].apply(map_stars_to_sentiment)

    # Tách văn bản và nhãn
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()

    # Chia thủ công train/test
    train_size = int(len(texts) * train_ratio)
    texts_train = texts[:train_size]
    labels_train = labels[:train_size]
    texts_test = texts[train_size:]
    labels_test = labels[train_size:]

    print(f"Số lượng mẫu trong train: {len(texts_train)}")
    print(f"Số lượng mẫu trong test: {len(texts_test)}")

    return texts_train, texts_test, labels_train, labels_test

# Hàm tokenize
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = text.strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return words

# Hàm tạo từ vựng
def build_vocab(texts, max_vocab_size=10000):
    all_tokens = [token for text in texts for token in tokenize(text)]
    token_counts = Counter(all_tokens)
    most_common_tokens = token_counts.most_common(max_vocab_size)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common_tokens)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab) + 1
    return vocab

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = [self.encode(text, vocab, max_len) for text in texts]
        self.labels = labels

    def encode(self, text, vocab, max_len):
        tokens = tokenize(text)
        encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        return encoded[:max_len] + [vocab["<PAD>"]] * (max_len - len(encoded))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

# Hàm tạo DataLoader
def create_dataloader(texts, labels, vocab, max_len=100, batch_size=32):
    dataset = ReviewDataset(texts, labels, vocab, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
