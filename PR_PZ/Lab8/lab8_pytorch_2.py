#%%
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from pathlib import Path
import jiwer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.transforms as T
from torchaudio.models import DeepSpeech
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
#%%
my_sample_rate = 22050
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=my_sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

# Аугментація
augmentation_transform = nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=50)
)

# train_transform = nn.Sequential(mel_spec, augmentation_transform) # тренувальні дані
# test_transform = mel_spectrogram_transform # тестові дані

class MyClassForLSSpeechDataset(Dataset):
    def __init__(self, csv_file_path, audio_dir, mel_transform, aug_transform=None):
        self.metadata_df = pd.read_csv(csv_file_path, sep='|', names=['filename', 'transcript', 'normalized'])
        self.audio_dir = audio_dir
        self.mel_transform = mel_transform
        self.aug_transform = aug_transform
        self.is_train = True

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        file_id = self.metadata_df.iloc[idx]['filename']
        transcript = str(self.metadata_df.iloc[idx]['normalized'])
        audio_file_path = os.path.join(self.audio_dir, f'{file_id}.wav')

        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
        except Exception as e:
            return None

        spectrogram = self.mel_transform(waveform)
        if self.is_train and self.aug_transform:
            spectrogram = self.aug_transform(spectrogram)

        return spectrogram, transcript

def text_to_indices(text):
    indices = []
    for char in text.lower():
        if char == ' ':
            indices.append(1)
        elif char in encoder_char_map:
            indices.append(encoder_char_map.find(char) + 2)
    return indices

def indices_to_text(indices_tensor, remove_padding=True):
    texts = []
    for row in indices_tensor:
        text = ""
        for idx in row:
            idx = idx.item()
            if remove_padding and idx == 0:
                continue
            if idx in (0, 1):
                text += " "
            elif idx > 1:
                try:
                    text += decoder_char_map[idx - 1]
                except IndexError:
                    text += "[?]"
        texts.append(text.strip())
    return texts


def my_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    spectrograms = []
    texts = []
    spec_lengths = []
    text_lengths = []

    for (spec, text) in batch:
        spectrograms.append(spec.squeeze(0).transpose(0, 1))
        spec_lengths.append(spec.shape[2])

        text_indices = torch.tensor(text_to_indices(text), dtype=torch.long)
        texts.append(text_indices)
        text_lengths.append(len(text_indices))

    spectrograms_padded = pad_sequence(spectrograms, batch_first=True, padding_value=0.0)
    spectrograms_padded = spectrograms_padded.transpose(1, 2)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0.0)

    spec_lengths = torch.tensor(spec_lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)

    return spectrograms_padded, texts_padded, spec_lengths, text_lengths

def greedy_decoder(output_tensor):
    best_path = torch.argmax(output_tensor, dim=2).transpose(0, 1)
    decoded_texts = []

    for sequence in best_path:
        decoded_sequence = []
        last_char_idx = -1

        for idx in sequence:
            idx = idx.item()
            if idx == last_char_idx:
                continue
            if idx != 0:
                decoded_sequence.append(idx)
            last_char_idx = idx

        text = ""
        for idx in decoded_sequence:
            if idx == 1:
                text += " "
            elif idx > 1:
                try:
                    text += decoder_char_map[idx - 1]
                except IndexError:
                    text += "[?]"
        decoded_texts.append(text)

    return decoded_texts

ljspeech_dataset = MyClassForLSSpeechDataset('data/LJSpeech-1.1/metadata.csv', 'data/LJSpeech-1.1/wavs', mel_spectrogram_transform)
encoder_char_map = "abcdefghijklmnopqrstuvwxyz"
decoder_char_map = " abcdefghijklmnopqrstuvwxyz"
#%%
train_size = int(0.95 * len(ljspeech_dataset))
val_size = len(ljspeech_dataset) - train_size

train_subset, val_subset = random_split(ljspeech_dataset, [train_size, val_size])
#%%
batch_size = 12
num_workers = 4

train_subset.dataset.is_train = True
train_loader = DataLoader(
    dataset=train_subset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=my_collate_fn,
    num_workers=num_workers
)

val_subset.dataset.is_train = False
val_loader = DataLoader(
    dataset=val_subset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=my_collate_fn,
    num_workers=num_workers
)
#%%
class DeepSpeech2(nn.Module):
    def __init__(self, n_features, n_class, n_cnn, n_rnn, rnn_hidden_size,
                 dropout, rnn_type='gru', bidirectional=True):
        super(DeepSpeech2, self).__init__()

        in_channels = 1
        cnn_channels = 32

        cnn_layers = []
        for i in range(n_cnn):
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=cnn_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1)
                )
            )
            cnn_layers.append(nn.ReLU())
            in_channels = cnn_channels

        self.cnn = nn.Sequential(*cnn_layers)
        rnn_input_size = n_features * cnn_channels

        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=n_rnn,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout if n_rnn > 1 else 0
            )
        else:
             self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=n_rnn,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout if n_rnn > 1 else 0
            )

        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size

        self.classifier = nn.Linear(
            in_features=rnn_output_size,
            out_features=n_class
        )

    def forward(self, x, input_lengths):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, -1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        x = F.log_softmax(x, dim=2)

        return x
#%%
n_features = 80
n_classes = 28
rnn_hidden_size = 512
n_rnn_layers = 5
n_cnn_layers = 2
dropout = 0.1

model = DeepSpeech2(
    n_features=n_features,
    n_class=n_classes,
    n_cnn=n_cnn_layers,
    n_rnn=n_rnn_layers,
    dropout=dropout,
    rnn_hidden_size=rnn_hidden_size,
    rnn_type='gru',
    bidirectional=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#%%
def train(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(
        loader,
        desc=f"Епоха {epoch+1} [Тренування]",
        unit="batch"
    )

    for i, batch in enumerate(loop):
        if batch is None:
            continue
        spectrograms, texts, spec_lengths, text_lengths = batch
        spectrograms = spectrograms.to(device)
        texts = texts.to(device)
        spec_lengths = spec_lengths.to(device)
        text_lengths = text_lengths.to(device)

        optimizer.zero_grad()
        output = model(spectrograms, spec_lengths)
        loss = criterion(output, texts, spec_lengths, text_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_epoch_loss = running_loss / len(loader)
    return avg_epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predicted_texts = []
    all_target_texts = []

    loop = tqdm(
        loader,
        desc="[Валідація]",
        unit="batch",
        leave=False
    )

    with torch.no_grad():
        for i, batch in enumerate(loop):
            if batch is None:
                continue

            spectrograms, texts, spec_lengths, text_lengths = batch
            spectrograms = spectrograms.to(device)
            texts = texts.to(device)
            spec_lengths = spec_lengths.to(device)
            text_lengths = text_lengths.to(device)

            output = model(spectrograms, spec_lengths)
            loss = criterion(output, texts, spec_lengths, text_lengths)
            running_loss += loss.item()

            predicted_texts = greedy_decoder(output)
            target_texts = indices_to_text(texts)
            all_predicted_texts.extend(predicted_texts)
            all_target_texts.extend(target_texts)

            loop.set_postfix(loss=f"{loss.item():.4f}")

    try:
        wer = jiwer.wer(all_target_texts, all_predicted_texts) * 100
        cer = jiwer.cer(all_target_texts, all_predicted_texts) * 100
    except ValueError as e:
        wer = float('nan')
        cer = float('nan')

    avg_val_loss = running_loss / len(loader)

    return avg_val_loss, wer, cer

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

optimizer = optim.AdamW(
    params=model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
#%%
num_epochs = 3

for epoch in range(num_epochs):

    avg_train_loss = train(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=epoch)

    avg_val_loss, wer, cer = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device)

    print(f"Середня Втрата (Тренування): {avg_train_loss:.4f}")
    print(f"Середня Втрата (Валідація): {avg_val_loss:.4f}")
    print(f"Character Error Rate (CER): {cer:.2f} %")
    print(f"Word Error Rate (WER): {wer:.2f} %")

#%%
save_model(model, 'DeepSpeech2_v0_3')
#%%
load_model(model, 'DeepSpeech2_v0_3')
#%%
avg_val_loss, wer, cer = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device)

print(f"Середня Втрата (Валідація): {avg_val_loss:.4f}")
print(f"Character Error Rate (CER): {cer:.2f} %")
print(f"Word Error Rate (WER): {wer:.2f} %")
#%%
def predict_single_file(
    model, audio_file_path, mel_transform,
    decoder_function, target_sample_rate, device):
    model.eval()

    try:
        # 2. Завантажуємо аудіо
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        print(f"Помилка завантаження файлу {audio_file_path}: {e}")
        return None

    waveform = waveform.to(device)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        ).to(device)
        waveform = resampler(waveform)

    mel_transform = mel_transform.to(device)
    spectrogram = mel_transform(waveform)
    spec_lengths = torch.tensor([spectrogram.shape[2]], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(spectrogram, spec_lengths)

    predicted_text = decoder_function(output)[0]

    return predicted_text
#%%
audio_file_paths = ['data/my_data/audio_1.m4a', 'data/my_data/audio_2.m4a', 'data/my_data/audio_3.m4a', 'data/my_data/audio_4.m4a']

for audio_file_path in audio_file_paths:
    try:
        recognized_text = predict_single_file(
            model=model,
            audio_file_path=audio_file_path,
            mel_transform=mel_spectrogram_transform,
            decoder_function=greedy_decoder,
            target_sample_rate=my_sample_rate,
            device=device
        )

        if recognized_text is not None:
            print(f"Файл: {audio_file_path}")
            print(f"Розпізнаний текст: {recognized_text}")

    except FileNotFoundError:
        print(f"Помилка: Тестовий файл не знайдено за шляхом {audio_file_path}")
