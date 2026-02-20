import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import math

# I rebuilt the entire model here for understandability
# you can import files fromt he training code to simple your work

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
            waveform = T.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mels(waveform)
        mel = self.db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        return mel.squeeze(0).transpose(0, 1)  # (T, 80)

# ===================== MODEL (EXACT MATCH WITH TRAINING) =====================

class ConvSubsampling(nn.Module):
    def __init__(self, input_dim=80, d_model=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2),
            nn.ReLU()
        )

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
        return x.view(B, T, C * F)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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


class CTCTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6, d_ff=1024):
        super().__init__()

        self.subsampling = ConvSubsampling(80, d_model)
        self.input_linear = nn.Linear(self.subsampling.output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.subsampling(x)
        x = self.input_linear(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# ===================== CTC GREEDY DECODER =====================

def ctc_greedy_decode(logits, idx2char, blank=0):
    pred_ids = torch.argmax(logits, dim=-1)

    decoded = []
    prev = blank

    for idx in pred_ids:
        idx = idx.item()
        if idx != blank and idx != prev:
            decoded.append(idx2char[idx])
        prev = idx

    return "".join(decoded)

# ===================== MAIN INFERENCE =====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load checkpoint
    ckpt = torch.load("ctc_asr_model.pt", map_location=device, weights_only=False)

    model = CTCTransformer(
        vocab_size=ckpt["config"]["vocab_size"],
        d_model=ckpt["config"]["d_model"],
        n_heads=ckpt["config"]["n_heads"],
        n_layers=ckpt["config"]["n_layers"],
        d_ff=ckpt["config"]["d_ff"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx2char = ckpt["vocab"]["idx2char"]
    audio_processor = MelSpectrogram()

    #use the test sample here
    # -> i used one audio from the training set
    audio_path = r"C:\Users\chapa mahindra\Downloads\1624-168623-0000.flac" 

    with torch.no_grad():
        mel = audio_processor(audio_path).unsqueeze(0).to(device)
        logits = model(mel)
        log_probs = torch.log_softmax(logits, dim=-1)
        text = ctc_greedy_decode(log_probs[0], idx2char)

    print("\n================ TRANSCRIPTION ================\n")
    print(text)
    print("\n===============================================\n")

if __name__ == "__main__":
    main()
