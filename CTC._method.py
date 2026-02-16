import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.transforms as T
import math
from pathlib import Path

# ===================== MEL SPECTROGRAM =====================

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, n_mels=80, hop_length=160):
        super().__init__()
        self.sample_rate = sample_rate
        self.mels = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length
        )
        self.db = T.AmplitudeToDB()

    def forward(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mels(waveform)
        mel_db = self.db(mel)

        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

        return mel_db.squeeze(0).transpose(0, 1)  # (T, 80)

# ===================== BUILD CHAR VOCAB =====================

def build_char_vocab(data_path):
    chars = set()
    data_path = Path(data_path)

    for speaker_dir in data_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue

            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"

            if trans_file.exists():
                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcript = parts[1].lower()
                            chars.update(list(transcript))

    chars = sorted(list(chars))
    char2idx = {c: i+1 for i, c in enumerate(chars)}  # 0 = blank
    idx2char = {i+1: c for i, c in enumerate(chars)}

    vocab_size = len(char2idx) + 1
    print("Vocabulary size (with blank):", vocab_size)

    return char2idx, idx2char, vocab_size

# ===================== DATASET =====================

class VoiceToTextDataset(Dataset):
    def __init__(self, data_path, char2idx, audio_processor):
        self.data = []
        self.char2idx = char2idx
        self.audio_processor = audio_processor

        data_path = Path(data_path)

        for speaker_dir in data_path.iterdir():
            if not speaker_dir.is_dir():
                continue

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue

                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_id, transcript = parts
                            audio_file = chapter_dir / f"{audio_id}.flac"

                            if audio_file.exists():
                                self.data.append({
                                    "audio_path": str(audio_file),
                                    "transcript": transcript.lower()
                                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mel = self.audio_processor(item["audio_path"])

        text_ids = [
            self.char2idx[c]
            for c in item["transcript"]
            if c in self.char2idx
        ]

        return mel, torch.tensor(text_ids, dtype=torch.long)

# ===================== COLLATE =====================

def collate_fn(batch):
    audio_batch = [b[0] for b in batch]
    text_batch = [b[1] for b in batch]

    audio_lengths = torch.tensor([a.shape[0] for a in audio_batch])
    text_lengths = torch.tensor([t.shape[0] for t in text_batch])

    audio_padded = pad_sequence(audio_batch, batch_first=True)
    text_padded = pad_sequence(text_batch, batch_first=True)

    return audio_padded, text_padded, audio_lengths, text_lengths

# ===================== CONV SUBSAMPLING =====================

class ConvSubsampling(nn.Module):
    def __init__(self, input_dim=80, d_model=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2),
            nn.ReLU()
        )

        # Compute output frequency dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, input_dim)
            out = self.conv(dummy)
            _, C, _, F = out.shape
            self.output_dim = C * F

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F)
        return x


# ===================== POSITIONAL ENCODING =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ===================== ENCODER LAYER =====================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

# ===================== CTC TRANSFORMER =====================

class CTCTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6, d_ff=1024):
        super().__init__()

        self.subsampling = ConvSubsampling(input_dim=80, d_model=d_model)
        self.input_linear = nn.Linear(self.subsampling.output_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.subsampling(x)
        x = self.input_linear(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        return self.output(x)

# ===================== LENGTH UPDATE =====================

def compute_subsampled_lengths(lengths):
    lengths = torch.floor((lengths - 3) / 2 + 1)
    lengths = torch.floor((lengths - 3) / 2 + 1)
    return lengths.long()

# ===================== TRAINING =====================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for audio, text, audio_lengths, text_lengths in loader:
        audio = audio.to(device)
        text = text.to(device)

        optimizer.zero_grad()

        logits = model(audio)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)

        input_lengths = compute_subsampled_lengths(audio_lengths).to(device)

        loss = criterion(
            log_probs,
            text,
            input_lengths,
            text_lengths.to(device)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ===================== MAIN =====================

if __name__ == "__main__":

    DATA_PATH = r"C:\Users\chapa mahindra\Downloads\train-clean-100\LibriSpeech\train-clean-100"
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char2idx, idx2char, vocab_size = build_char_vocab(DATA_PATH)

    audio_processor = MelSpectrogram()
    dataset = VoiceToTextDataset(DATA_PATH, char2idx, audio_processor)

    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=collate_fn)

    model = CTCTransformer(vocab_size).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
    torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": vocab_size,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 6,
        "d_ff": 1024
    },
    "vocab": {
        "char2idx": char2idx,
        "idx2char": idx2char
    },
    "version": "1.0.0"
}, "ctc_asr_model.pt")

print("Model saved successfully.")
