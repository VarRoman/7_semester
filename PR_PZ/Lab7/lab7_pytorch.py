#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import classification_report
import numpy as np
import tqdm
from collections import Counter
from torchtext.vocab import Vocab
from pathlib import Path
#%%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = get_tokenizer('basic_english')
print(device)
#%%
data_path = './yelp_review_polarity_csv/'

try:
    train_df = pd.read_csv(
        f"{data_path}train.csv",
        header=None,
        names=['label', 'text']
    )
    test_df = pd.read_csv(
        f"{data_path}test.csv",
        header=None,
        names=['label', 'text']
    )
    print("CSV файли успішно завантажено:")
    print(f"Навчання: {len(train_df)} рядків")
    print(f"Тестування: {len(test_df)} рядків")
    print(train_df.head())
except FileNotFoundError:
    print(f"Файли не знайдено у '{data_path}'")
#%%
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def yield_tokens(dataframe):
    for text in dataframe['text']:
        yield tokenizer(preprocess_text(text))

def print_train_time(start, end):
    total_time = end - start
    print(f"Загальний час тренування на {device}: {total_time:.3f} секунд")
    return total_time

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        if len(processed_text) > SEQUENCE_LENGTH:
            processed_text = processed_text[:SEQUENCE_LENGTH]
        else:
            processed_text.extend([pad_idx] * (SEQUENCE_LENGTH - len(processed_text)))

        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.float32)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    return label_list.to(device), text_list.to(device)

def train_step(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    pbar = tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}")

    for labels, text in pbar:
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)

        preds_binary = torch.round(torch.sigmoid(predictions))
        acc = (preds_binary == labels).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def test_step(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for labels, text in dataloader:
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)

            preds_binary = torch.round(torch.sigmoid(predictions))
            acc = (preds_binary == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.extend(preds_binary.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), all_labels, all_preds

def analyze_sentiment_pytorch(input_text, model, vocab, tokenizer, device):
    model.eval()
    cleaned = preprocess_text(input_text)
    tokenized = [vocab[token] for token in tokenizer(cleaned)]

    if len(tokenized) > SEQUENCE_LENGTH:
        tokenized = tokenized[:SEQUENCE_LENGTH]
    else:
        tokenized.extend([pad_idx] * (SEQUENCE_LENGTH - len(tokenized)))

    tensor = torch.tensor(tokenized, dtype=torch.int64).to(device)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor)

    prob = torch.sigmoid(prediction).item()
    sentiment = "Positive" if prob > 0.5 else "Negative"
    return f"{sentiment} (Ймовірність: {prob:.4f})"

def save_model(model: torch.nn.Module, model_name: str):
    model_path = Path("models")
    model_path.mkdir(parents=True,
                     exist_ok=True)

    model_full_name = f"{model_name}.pth"
    model_save_path = model_path / model_full_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    model_path = Path("models")
    model_full_name = f"{model_name}.pth"
    model_save_path = model_path / model_full_name
    model.load_state_dict(torch.load(f=model_save_path,weights_only=True))
    model = model.to(device)
    return model
#%%
print("Будуємо словник з train.csv")
counter = Counter()
for tokens in yield_tokens(train_df):
    counter.update(tokens)

vocab = Vocab(counter, specials=["<unk>", "<pad>"])

pad_idx = vocab['<pad>']
UNK_IDX = vocab['<unk>']
vocab_size = len(vocab)
SEQUENCE_LENGTH = 100

print(f"Розмір словника: {vocab_size}")
#%%
class YelpCSVDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx]['label']
        text = self.dataframe.iloc[idx]['text']
        return label, text

train_dataset = YelpCSVDataset(train_df)
test_dataset = YelpCSVDataset(test_df)
#%%
text_pipeline = lambda x: [vocab[token] for token in tokenizer(preprocess_text(x))]
label_pipeline = lambda x: 0 if x == 1 else 1

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)

#%%
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        last_hidden_state = hidden[-1, :, :]
        dropped = self.dropout(last_hidden_state)
        fc1_out = self.relu(self.fc1(dropped))
        output = self.fc2(fc1_out)
        return output

#%%
embedding_dim = 128
hidden_dim = 64
output_dim = 1
num_layers = 2
dropout = 0.5

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, pad_idx)

model = model.to(device)
model
#%%
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
NUM_EPOCHS = 5
#%%
print("Початок навчання:")
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_step(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc, _, _ = test_step(model, test_dataloader, criterion)

    print(f'\nЕпоха: {epoch+1:02}')
    print(f'\tНавчання: Втрати: {train_loss:.3f} | Точність: {train_acc*100:.2f}%')
    print(f'\tВалідація:  Втрати: {valid_loss:.3f} | Точність: {valid_acc*100:.2f}%')
#%%
print("Фінальна оцінка моделі:")
test_loss, test_acc, y_true, y_pred = test_step(model, test_dataloader, criterion)

print(f"Точність на тестових даних: {test_acc*100:.2f}%")
print("\nЗвіт про класифікацію:")
target_names = ['Negative (0)', 'Positive (1)']
print(classification_report(y_true, y_pred, target_names=target_names))
#%%
print("Тестування на нових відгуках")
sample_1 = "Absolutely loved the atmosphere and the food was fantastic!"
print(f"Відгук: {sample_1}\nНастрій: {analyze_sentiment_pytorch(sample_1, model, vocab, tokenizer, device)}\n")

sample_2 = "I had to wait forever and the service was terrible."
print(f"Відгук: {sample_2}\nНастрій: {analyze_sentiment_pytorch(sample_2, model, vocab, tokenizer, device)}\n")
#%%
save_model(model, 'LSTM_v0_2')
#%%
model = load_model(model, 'LSTM_v0_2')
#%%
