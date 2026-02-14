import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.transforms as T
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import math
from pathlib import Path

class AudioProcessor:
    """Converts audio to mel spectrogram features"""
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def process(self, audio_path):
        """Load audio and convert to mel spectrogram"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Transpose to (time, mel_bins) format
        return mel_spec_db.squeeze(0).transpose(0, 1)

class VoiceToTextDataset(Dataset):
    """Dataset for voice-to-text training using LibriSpeech format"""
    def __init__(self, data_path, tokenizer, audio_processor, max_audio_len=1000, max_text_len=200):
        self.data = []
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        # Load LibriSpeech data
        data_path = Path(data_path)
        
        # Traverse LibriSpeech directory structure
        for speaker_dir in data_path.iterdir():
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                
                # Find transcript file
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue
                
                # Read transcripts
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            audio_id, transcript = parts
                            audio_file = chapter_dir / f"{audio_id}.flac"
                            
                            if audio_file.exists():
                                self.data.append({
                                    'audio_path': str(audio_file),
                                    'transcript': transcript.strip()
                                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process audio
        mel_spec = self.audio_processor.process(item['audio_path'])
        
        # Truncate if too long
        if mel_spec.shape[0] > self.max_audio_len:
            mel_spec = mel_spec[:self.max_audio_len]
        
        # Tokenize text
        encoding = self.tokenizer.encode(item['transcript'])
        text_ids = encoding.ids[:self.max_text_len - 2]  # Reserve space for SOS/EOS
        text_ids = [2] + text_ids + [3]  # [SOS]=2, [EOS]=3
        
        return mel_spec, torch.tensor(text_ids, dtype=torch.long)

def build_text_tokenizer(data_path, tokenizer_path="voice_tokenizer.json"):
    """Build or load tokenizer for transcripts"""
    if os.path.exists(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)
    
    # Collect all transcripts
    sentences = []
    data_path = Path(data_path)
    
    for speaker_dir in data_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            if trans_file.exists():
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            sentences.append(parts[1])
    
    # Build tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
        vocab_size=5000
    )
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(tokenizer_path)
    
    return tokenizer

def collate_fn(batch):
    """Pad audio and text sequences to same length in batch"""
    audio_batch = [item[0] for item in batch]
    text_batch = [item[1] for item in batch]
    
    # Pad audio features (batch, time, n_mels)
    audio_lengths = torch.tensor([a.shape[0] for a in audio_batch])
    audio_padded = pad_sequence(audio_batch, batch_first=True, padding_value=0.0)
    
    # Pad text sequences
    text_padded = pad_sequence(text_batch, batch_first=True, padding_value=0)
    
    return audio_padded, text_padded, audio_lengths

#getting audio embeddings for chaging the dimension of audio features to match model dimension

class AudioEmbedding(nn.Module):
    """Converts mel spectrogram to model dimension"""
    def __init__(self, n_mels, d_model):
        super().__init__()
        self.linear = nn.Linear(n_mels, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # x: (batch, time, n_mels)
        return self.linear(x) * math.sqrt(self.d_model)

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross attention with encoder output
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

# ==================== VOICE TO TEXT TRANSFORMER ====================

class VoiceToTextTransformer(nn.Module):
    def __init__(self, n_mels, vocab_size, d_model=256, n_heads=4, n_layers=6, 
                 d_ff=1024, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Embeddings
        self.audio_embed = AudioEmbedding(n_mels, d_model)
        self.text_embed = TextEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, audio, audio_mask=None):
        x = self.audio_embed(audio)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, audio_mask)
        
        return x
    
    def decode(self, text, enc_output, src_mask=None, tgt_mask=None):
        x = self.text_embed(text)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, audio, text, audio_mask=None, text_mask=None):
        enc_output = self.encode(audio, audio_mask)
        dec_output = self.decode(text, enc_output, audio_mask, text_mask)
        logits = self.output_proj(dec_output)
        
        return logits

# ==================== TRAINING SETUP ====================

def create_masks(audio_lengths, text, pad_idx=0, device='cpu'):
    """Create attention masks for audio and text"""
    batch_size = text.size(0)
    max_audio_len = audio_lengths.max().item()
    max_text_len = text.size(1)
    
    # Audio mask (batch, 1, 1, audio_len) - create on correct device
    audio_mask = torch.arange(max_audio_len, device=device).expand(batch_size, max_audio_len) < audio_lengths.unsqueeze(1)
    audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
    
    # Text causal mask (no peeking ahead) - create on correct device
    text_mask = torch.tril(torch.ones(max_text_len, max_text_len, device=device)).bool()
    text_mask = text_mask.unsqueeze(0).unsqueeze(0)
    
    return audio_mask, text_mask

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (audio, text, audio_lengths) in enumerate(dataloader):
        audio = audio.to(device)
        text = text.to(device)
        audio_lengths = audio_lengths.to(device)
        
        # Prepare input and target
        text_input = text[:, :-1]
        text_target = text[:, 1:]
        
        # Create masks
        audio_mask, text_mask = create_masks(audio_lengths, text_input, device=device)
        
        # Forward pass
        logits = model(audio, text_input, audio_mask, text_mask)
        
        # Calculate loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), text_target.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

# ==================== MAIN ====================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = r"C:\Users\chapa mahindra\Downloads\train-clean-100\LibriSpeech\train-clean-100"
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 3e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = build_text_tokenizer(DATA_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    print("Loading dataset...")
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
    dataset = VoiceToTextDataset(DATA_PATH, tokenizer, audio_processor)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Initialize model
    print("Initializing model...")
    model = VoiceToTextTransformer(
        n_mels=80,
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,  # Reduced for faster training
        d_ff=1024,
        dropout=0.1
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CTCLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch {epoch}/{EPOCHS} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'tokenizer_path': 'voice_tokenizer.json',
        'config': {
            'n_mels': 80,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 4,
            'd_ff': 1024
        }
    }, 'voice_to_text_model.pth')
    
    print("Training complete! Model saved to voice_to_text_model.pth")
