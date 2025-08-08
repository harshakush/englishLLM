# Create the refactored code with enhanced logging, colorful output, and improved checkpointing
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
import glob
import argparse
import logging
import matplotlib.pyplot as plt
import re
import json
import time
from datetime import datetime, timedelta
import sys

# ========== COLORFUL LOGGING SETUP ==========
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\\033[36m',     # Cyan
        'INFO': '\\033[32m',      # Green
        'WARNING': '\\033[33m',   # Yellow
        'ERROR': '\\033[31m',     # Red
        'CRITICAL': '\\033[35m',  # Magenta
        'RESET': '\\033[0m',      # Reset
        'BOLD': '\\033[1m',       # Bold
        'BLUE': '\\033[34m',      # Blue
        'PURPLE': '\\033[95m',    # Purple
    }
    
    def format(self, record):
        # Add color to levelname
        levelname_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{levelname_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Add color to timestamp
        log_time = self.formatTime(record, self.datefmt)
        colored_time = f"{self.COLORS['BLUE']}{log_time}{self.COLORS['RESET']}"
        
        # Format the message
        if hasattr(record, 'color'):
            message = f"{record.color}{record.getMessage()}{self.COLORS['RESET']}"
        else:
            message = record.getMessage()
            
        return f"{colored_time} [{record.levelname}] {message}"

# ========== OUTPUT DIRECTORY ==========
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== ENHANCED LOGGING SETUP ==========
def setup_logging():
    """Setup enhanced logging with colors and detailed formatting"""
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = ColoredFormatter(
        datefmt="%H:%M:%S"
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler for colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ========== PROGRESS TRACKING ==========
class TrainingProgress:
    """Class to track and save training progress"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "training_progress.json")
        self.start_time = time.time()
        self.epoch_times = []
        
    def save_progress(self, epoch, train_loss, val_loss, val_ppl, best_val_loss, 
                     train_losses, val_losses, val_ppls, model_params):
        """Save detailed progress information"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = model_params['num_epochs'] - (epoch + 1)
            eta = remaining_epochs * avg_epoch_time
        else:
            avg_epoch_time = 0
            eta = 0
            
        progress_data = {
            'current_epoch': epoch,
            'total_epochs': model_params['num_epochs'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_ppl,
            'best_val_loss': best_val_loss,
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'average_epoch_time': avg_epoch_time,
            'eta_seconds': eta,
            'eta_formatted': str(timedelta(seconds=int(eta))),
            'timestamp': datetime.now().isoformat(),
            'train_losses_history': train_losses,
            'val_losses_history': val_losses,
            'val_perplexity_history': val_ppls,
            'model_parameters': model_params,
            'epoch_times': self.epoch_times
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def load_progress(self):
        """Load progress from file if exists"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None
        
    def add_epoch_time(self, epoch_time):
        """Add epoch time for ETA calculation"""
        self.epoch_times.append(epoch_time)

# ========== THREADING FOR DATA LOADING ==========
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ========== DATA CLEANING ==========
def clean_text(text):
    """Enhanced text cleaning with detailed logging"""
    original_length = len(text)
    
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'#+', '', text)  # Remove markdown headers
    text = re.sub(r'\\[.*?\\]\\(.*?\\)', '', text)  # Remove markdown links
    text = re.sub(r'[*_~]', '', text)  # Remove markdown formatting
    text = re.sub(r'[\\[\\]<>]', '', text)  # Remove brackets
    text = re.sub(r'\\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    cleaned_length = len(text)
    reduction_percent = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
    
    if reduction_percent > 50:
        logger.debug(f"High text reduction: {reduction_percent:.1f}% (from {original_length} to {cleaned_length} chars)")
    
    return text

def clean_sentences(sentences):
    """Clean sentences with progress tracking"""
    logger.info(f"🧹 Cleaning {len(sentences):,} sentences...")
    
    cleaned = []
    empty_sentences = 0
    
    for i, sent in enumerate(tqdm(sentences, desc="Cleaning sentences", disable=len(sentences) < 10000)):
        sent_str = ' '.join(sent)
        sent_str = clean_text(sent_str)
        words = [w for w in sent_str.split() if w]
        
        if words:
            cleaned.append(words)
        else:
            empty_sentences += 1
            
        if i % 50000 == 0 and i > 0:
            logger.debug(f"Processed {i:,} sentences, {empty_sentences} empty after cleaning")
    
    logger.info(f"✅ Cleaning complete: {len(cleaned):,} sentences retained, {empty_sentences:,} empty sentences removed")
    return cleaned

# ========== DATA PREP ==========
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len
        logger.debug(f"Created TextDataset with {len(token_ids):,} tokens, seq_len={seq_len}")

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

def build_vocab(sentences, min_freq=2):
    """Build vocabulary with detailed statistics"""
    logger.info(f"🔤 Building vocabulary with min_freq={min_freq}...")
    
    counter = Counter()
    total_words = 0
    
    for sent in tqdm(sentences, desc="Counting words", disable=len(sentences) < 10000):
        counter.update(sent)
        total_words += len(sent)
    
    # Filter by frequency
    vocab_filtered = [w for w, c in counter.items() if c >= min_freq]
    vocab = ['<pad>', '<unk>'] + vocab_filtered
    
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    # Statistics
    total_unique_words = len(counter)
    vocab_size = len(vocab)
    coverage = sum(c for w, c in counter.items() if c >= min_freq) / total_words * 100
    
    logger.info(f"📊 Vocabulary Statistics:")
    logger.info(f"   Total words: {total_words:,}")
    logger.info(f"   Unique words: {total_unique_words:,}")
    logger.info(f"   Vocabulary size (after filtering): {vocab_size:,}")
    logger.info(f"   Coverage: {coverage:.2f}%")
    logger.info(f"   Most common words: {counter.most_common(10)}")
    
    return vocab, word2idx, idx2word

def tokenize(sentences, word2idx):
    """Tokenize sentences with OOV tracking"""
    logger.info("🔢 Tokenizing sentences...")
    
    tokens = []
    oov_count = 0
    total_tokens = 0
    
    for sent in tqdm(sentences, desc="Tokenizing", disable=len(sentences) < 10000):
        for w in sent:
            token_id = word2idx.get(w, word2idx['<unk>'])
            tokens.append(token_id)
            total_tokens += 1
            if token_id == word2idx['<unk>']:
                oov_count += 1
    
    oov_rate = (oov_count / total_tokens * 100) if total_tokens > 0 else 0
    logger.info(f"✅ Tokenization complete: {len(tokens):,} tokens, OOV rate: {oov_rate:.2f}%")
    
    return tokens

# ========== MODEL ==========
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, emb_size=256, nhead=8, num_layers=6, dim_feedforward=1024, max_seq_len=32, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(max_seq_len, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.max_seq_len = max_seq_len
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"🏗️  Model Architecture:")
        logger.info(f"   Vocabulary size: {vocab_size:,}")
        logger.info(f"   Embedding size: {emb_size}")
        logger.info(f"   Number of heads: {nhead}")
        logger.info(f"   Number of layers: {num_layers}")
        logger.info(f"   Feedforward dimension: {dim_feedforward}")
        logger.info(f"   Max sequence length: {max_seq_len}")
        logger.info(f"   Dropout: {dropout}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer(x, mask=mask)
        logits = self.fc_out(x)
        return logits

# ========== HUGGINGFACE WRAPPERS ==========
from transformers import PreTrainedModel, PretrainedConfig

class CustomTransformerConfig(PretrainedConfig):
    model_type = "custom-transformer"

    def __init__(
        self,
        vocab_size=30522,
        emb_size=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        max_seq_len=32,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.dropout = dropout

class CustomTransformerLM(PreTrainedModel):
    config_class = CustomTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerLM(
            vocab_size=config.vocab_size,
            emb_size=config.emb_size,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)

# ========== ENHANCED CHECKPOINTING ==========
def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path, 
                   train_losses, val_losses, val_ppls, progress_tracker):
    """Enhanced checkpoint saving with more metadata"""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ppls': val_ppls,
        'timestamp': datetime.now().isoformat(),
        'epoch_times': progress_tracker.epoch_times,
        'total_training_time': time.time() - progress_tracker.start_time
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    
    # Also save a backup checkpoint
    backup_path = checkpoint_path.replace('.pt', f'_epoch_{epoch+1}.pt')
    torch.save(checkpoint_data, backup_path)
    
    logger.info(f"💾 Checkpoint saved at epoch {epoch+1} (backup: {os.path.basename(backup_path)})")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Enhanced checkpoint loading with validation"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
        return 0, float('inf'), [], [], []
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_ppls = checkpoint.get('val_ppls', [])
        
        logger.info(f"✅ Checkpoint loaded from epoch {epoch+1}")
        logger.info(f"   Best validation loss: {best_val_loss:.4f}")
        logger.info(f"   Training history: {len(train_losses)} epochs")
        
        return epoch, best_val_loss, train_losses, val_losses, val_ppls
        
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        return 0, float('inf'), [], [], []

def should_pause():
    return os.path.exists("PAUSE.TXT")

# ========== ENHANCED TRAINING LOOP ==========
def train_epoch(model, loader, optimizer, criterion, device, epoch, best_val_loss, 
               checkpoint_path, progress_tracker, train_losses, val_losses, val_ppls):
    """Enhanced training epoch with detailed logging"""
    model.train()
    total_loss = 0
    num_batches = len(loader)
    epoch_start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(loader, desc=f"🚂 Epoch {epoch+1} Training", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, (x, y) in enumerate(pbar):
        batch_start_time = time.time()
        
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * x.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{batch_loss:.4f}',
            'Avg': f'{total_loss / ((batch_idx + 1) * loader.batch_size):.4f}'
        })
        
        # Log detailed batch info every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            batch_time = time.time() - batch_start_time
            logger.debug(f"Batch {batch_idx}/{num_batches}: loss={batch_loss:.4f}, time={batch_time:.3f}s")
        
        # Check for pause signal
        if should_pause():
            logger.warning("⏸️  Pause signal detected. Saving checkpoint and exiting...")
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                          train_losses, val_losses, val_ppls, progress_tracker)
            logger.info("You can now safely restart your machine. To resume, run with --resume.")
            exit(0)
    
    pbar.close()
    epoch_time = time.time() - epoch_start_time
    progress_tracker.add_epoch_time(epoch_time)
    
    avg_loss = total_loss / len(loader.dataset)
    logger.info(f"📈 Training epoch {epoch+1} completed in {epoch_time:.1f}s - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

def eval_epoch(model, loader, criterion, device, epoch):
    """Enhanced evaluation epoch with detailed logging"""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    pbar = tqdm(loader, desc=f"🔍 Epoch {epoch+1} Validation", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            num_samples += x.size(0)
            
            pbar.set_postfix({'Val Loss': f'{batch_loss:.4f}'})
    
    pbar.close()
    avg_loss = total_loss / num_samples
    logger.info(f"📊 Validation epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

def calculate_perplexity(loss):
    """Calculate perplexity with bounds checking"""
    try:
        ppl = math.exp(loss)
        if ppl > 1e6:  # Cap extremely high perplexity values
            logger.warning(f"⚠️  Very high perplexity: {ppl:.2e}")
        return ppl
    except OverflowError:
        logger.warning("⚠️  Perplexity overflow, returning inf")
        return float('inf')

# ========== ENHANCED VISUALIZATION ==========
def plot_metrics(train_losses, val_losses, val_ppls, output_dir):
    """Enhanced plotting with better styling"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Perplexity plot
    ax2.plot(epochs, val_ppls, label='Val Perplexity', color='orange', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Validation Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss difference plot
    if len(train_losses) > 1:
        loss_diff = [val_losses[i] - train_losses[i] for i in range(len(train_losses))]
        ax3.plot(epochs, loss_diff, label='Val - Train Loss', color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.set_title('Overfitting Monitor (Val - Train Loss)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Learning curve (log scale)
    ax4.semilogy(epochs, train_losses, label='Train Loss (log)', color='blue', linewidth=2)
    ax4.semilogy(epochs, val_losses, label='Val Loss (log)', color='red', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Learning Curves (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"📊 Training curves saved to {plot_path}")

# ========== MAIN SCRIPT ==========
def main():
    parser = argparse.ArgumentParser(description="Enhanced Transformer Language Model Training")
    parser.add_argument('--pause', action='store_true', help='Pause training at the end of the next batch')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--restart', action='store_true', help='Restart training from scratch')
    parser.add_argument('--data-path', type=str, default="/home/harsha/Dev/englishLLM/data", 
                       help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('- ', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    args = parser.parse_args()

    # Print startup banner
    logger.info("=" * 80)
    logger.info("🚀 ENHANCED TRANSFORMER LANGUAGE MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    
    # --- Hyperparameters ---
    seq_len = 64
    batch_size = 32
    emb_size = 64
    nhead = 8
    num_layers = 16
    dim_feedforward = 64
    dropout = 0.1
    num_epochs = 10
    lr = args.lr
    min_freq = 2
    max_seq_len = seq_len
    num_workers = 2
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
    best_model_path = os.path.join(OUTPUT_DIR, "best_transformer_lm.pt")

    # Initialize progress tracker
    progress_tracker = TrainingProgress(OUTPUT_DIR)
    
    # Log hyperparameters
    model_params = {
        'seq_len': seq_len, 'batch_size': batch_size, 'emb_size': emb_size,
        'nhead': nhead, 'num_layers': num_layers, 'dim_feedforward': dim_feedforward,
        'dropout': dropout, 'num_epochs': num_epochs, 'lr': lr, 'min_freq': min_freq
    }
    
    logger.info("⚙️  Hyperparameters:")
    for key, value in model_params.items():
        logger.info(f"   {key}: {value}")

    # --- Data Loading from local folder ---
    logger.info(f"📂 Loading data from: {args.data_path}")
    all_filenames = [f for f in glob.glob(os.path.join(args.data_path, "*")) if os.path.isfile(f)]
    
    if not all_filenames:
        logger.error(f"❌ No files found in {args.data_path}")
        return
    
    logger.info(f"📄 Found {len(all_filenames)} files:")
    for i, filename in enumerate(all_filenames[:10]):  # Show first 10 files
        size_mb = os.path.getsize(filename) / 1024 / 1024
        logger.info(f"   {i+1}. {os.path.basename(filename)} ({size_mb:.1f} MB)")
    if len(all_filenames) > 10:
        logger.info(f"   ... and {len(all_filenames) - 10} more files")
    
    # Load sentences
    sentences = []
    total_lines = 0
    
    for filename in tqdm(all_filenames, desc="Loading files"):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_sentences = 0
                for line in f:
                    words = [w.lower() for w in line.strip().split() if w.strip()]
                    if words:
                        sentences.append(words)
                        file_sentences += 1
                        total_lines += 1
                logger.debug(f"Loaded {file_sentences:,} sentences from {os.path.basename(filename)}")
        except Exception as e:
            logger.error(f"❌ Error loading {filename}: {e}")
    
    logger.info(f"✅ Loaded {len(sentences):,} sentences from {len(all_filenames)} files")

    # --- Data Cleaning ---
    sentences = clean_sentences(sentences)

    # --- Build Vocab ---
    vocab, word2idx, idx2word = build_vocab(sentences, min_freq=min_freq)
    vocab_size = len(vocab)

    # --- Tokenize ---
    tokens = tokenize(sentences, word2idx)
    split = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    
    logger.info(f"📊 Data split: {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens")

    # --- Device selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🖥️  Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("🖥️  Using CPU")
    
    pin_memory = True if device.type == 'cuda' else False

    # --- Datasets and Loaders ---
    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    
    logger.info(f"📦 Dataset sizes: {len(train_dataset):,} train samples, {len(val_dataset):,} val samples")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory
    )

    # --- Model, Optimizer, Loss ---
    model = TransformerLM(vocab_size, emb_size, nhead, num_layers, dim_feedforward, max_seq_len, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

    # --- Checkpoint Handling ---
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_ppls = []
    
    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss, train_losses, val_losses, val_ppls = load_checkpoint(
            model, optimizer, checkpoint_path)
        start_epoch += 1
        logger.info(f"🔄 Resuming from epoch {start_epoch}")
    elif args.restart:
        logger.info("🔄 Restarting training from scratch")
        start_epoch = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        val_ppls = []

    # --- Training Loop ---
    logger.info("🎯 Starting training loop...")
    logger.info("=" * 80)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            logger.info(f"\\n🔄 EPOCH {epoch+1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                   epoch, best_val_loss, checkpoint_path, progress_tracker,
                                   train_losses, val_losses, val_ppls)
            
            # Validation
            val_loss = eval_epoch(model, val_loader, criterion, device, epoch)
            val_ppl = calculate_perplexity(val_loss)
            
            # Update metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_ppls.append(val_ppl)
            
            # Check for best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logger.info("🏆 New best model saved!")
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\\n📊 EPOCH {epoch+1} SUMMARY:")
            logger.info(f"   Train Loss: {train_loss:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f} {'🏆 (BEST!)' if is_best else ''}")
            logger.info(f"   Val Perplexity: {val_ppl:.2f}")
            logger.info(f"   Epoch Time: {epoch_time:.1f}s")
            logger.info(f"   Best Val Loss: {best_val_loss:.4f}")
            
            # Save progress
            progress_tracker.save_progress(
                epoch, train_loss, val_loss, val_ppl, best_val_loss,
                train_losses, val_losses, val_ppls, model_params
            )
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                          train_losses, val_losses, val_ppls, progress_tracker)
            
            if args.pause:
                logger.info("⏸️  Pausing training as requested by --pause.")
                break
                
    except KeyboardInterrupt:
        logger.info("\\n⏹️  Training interrupted by user")
        save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path,
                      train_losses, val_losses, val_ppls, progress_tracker)
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        raise

    logger.info("\\n🎉 Training completed!")
    logger.info("=" * 80)

    # --- Visualization ---
    if train_losses:
        plot_metrics(train_losses, val_losses, val_ppls, OUTPUT_DIR)

    # --- Save HuggingFace format ---
    logger.info("💾 Saving HuggingFace format...")
    hf_dir = os.path.join(OUTPUT_DIR, "hf_model")
    os.makedirs(hf_dir, exist_ok=True)

    hf_config = CustomTransformerConfig(
        vocab_size=vocab_size, emb_size=emb_size, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len, dropout=dropout
    )

    hf_model = CustomTransformerLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict())
    hf_model.save_pretrained(hf_dir)
    hf_config.save_pretrained(hf_dir)

    # Save vocabulary and index mappings
    with open(os.path.join(hf_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(hf_dir, "idx2word.json"), "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ HuggingFace model and vocab saved to {hf_dir}")

    # --- Generation Example ---
    def generate(seed, max_new_tokens=30, temperature=1.0):
        """Enhanced generation function with better handling"""
        model.eval()
        tokens = [word2idx.get(w, word2idx['<unk>']) for w in seed]
        
        logger.debug(f"Generating with seed: {seed}, max_tokens: {max_new_tokens}, temp: {temperature}")
        
        for i in range(max_new_tokens):
            # Use last seq_len tokens or pad if shorter
            input_tokens = tokens[-seq_len:] if len(tokens) >= seq_len else ([word2idx['<pad>']] * (seq_len - len(tokens)) + tokens)
            x = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(x)
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
            
            # Stop at sentence end
            if idx2word[next_token] in ['.', '!', '?']:
                break
                
        return ' '.join([idx2word[t] for t in tokens])

    # Interactive generation
    logger.info("\\n🎮 Interactive generation mode activated!")
    logger.info("💡 Enter your prompt (words separated by spaces)")
    logger.info("💡 Press Ctrl+C or Ctrl+D to exit")
    logger.info("-" * 50)

    try:
        while True:
            try:
                user_input = input("\\n🎯 Prompt: ").strip()
                if not user_input:
                    logger.warning("⚠️  Please enter a non-empty prompt.")
                    continue
                    
                seed = user_input.lower().split()

                try:
                    max_new_tokens = int(input("📏 Max new tokens [20]: ").strip() or 20)
                except ValueError:
                    logger.warning("⚠️  Invalid input. Using default value 20.")
                    max_new_tokens = 20

                try:
                    temperature = float(input("🌡️  Temperature [0.8]: ").strip() or 0.8)
                except ValueError:
                    logger.warning("⚠️  Invalid input. Using default value 0.8.")
                    temperature = 0.8

                logger.info("🤖 Generating...")
                output = generate(seed, max_new_tokens=max_new_tokens, temperature=temperature)
                
                print("\\n" + "="*60)
                print(f"🎯 Input: {user_input}")
                print(f"🤖 Generated: {output}")
                print("="*60)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
    
    logger.info("\\n👋 Goodbye! Thanks for using the Enhanced Transformer LM!")
    logger.info(f"📊 Final training summary saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
