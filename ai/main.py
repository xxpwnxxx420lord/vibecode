"""
Self-Training AI Bot
Trains a GPT-style transformer on text files in the ./data/ directory.
Supports GPU acceleration via CUDA.

Structure:
    project/
    ├── data/       <- drop .txt files here
    └── main.py
"""

import os
import math
import time
import pickle
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    vocab_size: int = 256           # set at runtime from tokenizer
    block_size: int = 512           # bigger context = smarter replies
    n_embd: int = 512               # wider embedding
    n_heads: int = 8                # more attention heads
    n_layers: int = 8               # deeper network
    dropout: float = 0.15           # light regularization for large data
    bias: bool = True


@dataclass
class TrainConfig:
    data_dir: str = str(Path(__file__).parent / "data")
    checkpoint_dir: str = str(Path(__file__).parent / "checkpoints")
    batch_size: int = 48            # sweet spot for 8.6GB VRAM at this model size
    learning_rate: float = 2e-4     # slightly lower = more stable for bigger model
    weight_decay: float = 0.1
    max_epochs: int = 10            # large dataset needs fewer epochs to converge
    eval_interval: int = 1000       # less frequent eval = faster training
    save_interval: int = 2000
    grad_clip: float = 1.0
    warmup_steps: int = 500         # longer warmup for bigger model
    compile_model: bool = False
    mixed_precision: bool = True    # fp16 on CUDA — big speed boost
    num_workers: int = 0            # 0 is safer on Windows, avoids multiprocessing issues
    resume: bool = True


# ──────────────────────────────────────────────
# Tokenizer (character-level, upgradeable)
# ──────────────────────────────────────────────
class CharTokenizer:
    def __init__(self):
        self.stoi: dict = {}
        self.itos: dict = {}
        self.vocab_size: int = 0

    def fit(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        log.info(f"Vocab size: {self.vocab_size} unique chars")

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.stoi = data["stoi"]
        self.itos = data["itos"]
        self.vocab_size = len(self.stoi)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def load_data(data_dir: str, tokenizer: CharTokenizer, block_size: int = 256) -> Tuple[TextDataset, TextDataset]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data/ directory not found at {data_dir}")

    texts = []
    for ext in ("*.txt", "*.md", "*.csv", "*.json", "*.py"):
        for p in sorted(data_path.rglob(ext)):
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                log.info(f"  Loaded: {p} ({p.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                log.warning(f"  Skip {p}: {e}")

    if not texts:
        raise ValueError(f"No readable text files found in {data_dir}")

    corpus = "\n\n".join(texts)
    log.info(f"Total corpus: {len(corpus):,} chars")

    tokenizer.fit(corpus)
    tokens = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)

    # 90/10 train/val split
    split = int(0.9 * len(tokens))
    train_ds = TextDataset(tokens[:split], block_size=block_size)
    val_ds = TextDataset(tokens[split:], block_size=block_size)
    log.info(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    return train_ds, val_ds


# ──────────────────────────────────────────────
# Model Components
# ──────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_heads

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, hidden, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(hidden, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        log.info(f"Model parameters: {self.num_params():,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence too long: {T} > {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        model: MiniGPT,
        train_ds: TextDataset,
        val_ds: TextDataset,
        tokenizer: CharTokenizer,
        tcfg: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tokenizer = tokenizer
        self.tcfg = tcfg
        self.device = device

        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg.max_epochs,
            eta_min=tcfg.learning_rate / 10,
        )
        self.scaler = GradScaler("cuda", enabled=(tcfg.mixed_precision and device.type == "cuda"))
        self.step = 0
        self.best_val_loss = float("inf")

        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

        if tcfg.resume:
            self._load_latest_checkpoint()

    def _build_optimizer(self) -> AdamW:
        # Separate weight decay params
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)

        groups = [
            {"params": decay, "weight_decay": self.tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(groups, lr=self.tcfg.learning_rate, betas=(0.9, 0.95))

    def _make_loader(self, ds: TextDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.tcfg.batch_size,
            shuffle=shuffle,
            num_workers=self.tcfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

    @torch.no_grad()
    def estimate_loss(self, n_batches: int = 20) -> dict:
        self.model.eval()
        results = {}
        for split, ds in [("train", self.train_ds), ("val", self.val_ds)]:
            loader = self._make_loader(ds, shuffle=True)
            losses = []
            for i, (x, y) in enumerate(loader):
                if i >= n_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                with autocast("cuda", enabled=self.tcfg.mixed_precision and self.device.type == "cuda"):
                    _, loss = self.model(x, y)
                losses.append(loss.item())
            results[split] = sum(losses) / len(losses)
        self.model.train()
        return results

    def _save_checkpoint(self, tag: str = "latest"):
        path = os.path.join(self.tcfg.checkpoint_dir, f"ckpt_{tag}.pt")
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        log.info(f"Checkpoint saved → {path}")

    def _load_latest_checkpoint(self):
        path = os.path.join(self.tcfg.checkpoint_dir, "ckpt_latest.pt")
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt["best_val_loss"]
        log.info(f"Resumed from step {self.step} (best val loss: {self.best_val_loss:.4f})")

    def _sample(self, prompt: str = "\n", n_tokens: int = 120) -> str:
        self.model.eval()
        context = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        out = self.model.generate(context, max_new_tokens=n_tokens)
        self.model.train()
        return self.tokenizer.decode(out[0].tolist())

    def train(self):
        log.info(f"Training on device: {self.device}")
        train_loader = self._make_loader(self.train_ds, shuffle=True)

        for epoch in range(1, self.tcfg.max_epochs + 1):
            epoch_loss = 0.0
            t0 = time.time()

            for batch_idx, (x, y) in enumerate(train_loader):
                # Warmup LR
                if self.step < self.tcfg.warmup_steps:
                    lr = self.tcfg.learning_rate * (self.step + 1) / self.tcfg.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                x, y = x.to(self.device), y.to(self.device)

                with autocast("cuda", enabled=self.tcfg.mixed_precision and self.device.type == "cuda"):
                    _, loss = self.model(x, y)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                self.step += 1

                # Eval
                if self.step % self.tcfg.eval_interval == 0:
                    losses = self.estimate_loss()
                    log.info(
                        f"Step {self.step:>6} | "
                        f"train loss: {losses['train']:.4f} | "
                        f"val loss: {losses['val']:.4f}"
                    )
                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                        self._save_checkpoint("best")
                    sample = self._sample()
                    log.info(f"\n── Sample ──\n{sample[:300]}\n───────────")

                # Save
                if self.step % self.tcfg.save_interval == 0:
                    self._save_checkpoint("latest")

            elapsed = time.time() - t0
            avg_loss = epoch_loss / (batch_idx + 1)
            log.info(
                f"Epoch {epoch:>3}/{self.tcfg.max_epochs} | "
                f"avg loss: {avg_loss:.4f} | "
                f"time: {elapsed:.1f}s | "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            self.scheduler.step()

        self._save_checkpoint("final")
        log.info("Training complete.")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        log.info("Using Apple MPS (Metal)")
    else:
        dev = torch.device("cpu")
        log.warning("No GPU found — training on CPU (slow)")
    return dev


def load_model_from_checkpoint(device: torch.device, block_size: int, n_embd: int, n_heads: int, n_layers: int):
    """Load tokenizer + model from the best available checkpoint. Returns (model, tokenizer) or (None, None)."""
    ckpt_dir = Path(__file__).parent / "checkpoints"
    tok_path = ckpt_dir / "tokenizer.pkl"
    ckpt_path = ckpt_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "ckpt_latest.pt"

    if not tok_path.exists() or not ckpt_path.exists():
        return None, None

    tokenizer = CharTokenizer()
    tokenizer.load(str(tok_path))

    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model = MiniGPT(mcfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded checkpoint: {ckpt_path.name} (step {ckpt.get('step', '?')})")
    return model, tokenizer


def chat_loop(model: MiniGPT, tokenizer: CharTokenizer, device: torch.device,
              temperature: float, top_k: int, max_tokens: int):
    """Interactive chat REPL."""
    print("\n" + "═" * 50)
    print("  4chan Bot — Chat Mode")
    print("  Type your message and press Enter.")
    print("  Commands: :temp <0.1-2.0>  :tokens <n>  :quit")
    print("═" * 50 + "\n")

    # Rolling conversation context so replies build on each other
    context_tokens: list[int] = []
    max_ctx = model.cfg.block_size

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: later anon o7")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.startswith(":quit"):
            print("Bot: later anon o7")
            break

        if user_input.startswith(":temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[temperature set to {temperature}]")
            except ValueError:
                print("[invalid value]")
            continue

        if user_input.startswith(":tokens "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"[max tokens set to {max_tokens}]")
            except ValueError:
                print("[invalid value]")
            continue

        # ── Encode user message and append to context ──
        user_enc = tokenizer.encode(f"\n[USER] {user_input}\n[BOT] ")
        context_tokens.extend(user_enc)

        # Trim context to block_size
        if len(context_tokens) > max_ctx:
            context_tokens = context_tokens[-max_ctx:]

        # ── Generate ──
        ctx_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model.generate(
                ctx_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode only the newly generated tokens
        new_tokens = out[0][len(context_tokens):].tolist()
        reply = tokenizer.decode(new_tokens).strip()

        # Trim reply at natural stopping points
        for stop in ["\n[USER]", "\n[BOT]", "\n[THREAD", "[OP #"]:
            if stop in reply:
                reply = reply[:reply.index(stop)].strip()

        print(f"\nBot: {reply}\n")

        # Add bot reply to rolling context
        context_tokens.extend(tokenizer.encode(reply + "\n"))
        if len(context_tokens) > max_ctx:
            context_tokens = context_tokens[-max_ctx:]


def main():
    parser = argparse.ArgumentParser(description="Self-training GPT on local data/")
    parser.add_argument("--data_dir", default=str(Path(__file__).parent / "data"), help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--no_resume", action="store_true", help="Start fresh")
    parser.add_argument("--compile", action="store_true", help="torch.compile (PyTorch 2.0+)")
    parser.add_argument("--generate", type=str, default=None, help="Generate text from a prompt (one-shot)")
    parser.add_argument("--chat", action="store_true", help="Launch interactive chat mode")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling (default: 40)")
    parser.add_argument("--max_tokens", type=int, default=300, help="Max tokens per reply (default: 300)")
    args = parser.parse_args()

    device = get_device()

    # ── Chat mode — no training needed ──
    if args.chat:
        model, tokenizer = load_model_from_checkpoint(
            device, args.block_size, args.n_embd, args.n_heads, args.n_layers
        )
        if model is None:
            print("No checkpoint found. Train the model first with: py main.py")
            return
        chat_loop(model, tokenizer, device,
                  temperature=args.temperature,
                  top_k=args.top_k,
                  max_tokens=args.max_tokens)
        return

    # ── Load data (needed for training and one-shot generate) ──
    tokenizer = CharTokenizer()
    log.info(f"Loading data from: {args.data_dir}")
    train_ds, val_ds = load_data(args.data_dir, tokenizer, block_size=args.block_size)

    ckpt_dir = str(Path(__file__).parent / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save(os.path.join(ckpt_dir, "tokenizer.pkl"))

    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = MiniGPT(mcfg).to(device)

    if args.compile:
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── One-shot generate ──
    if args.generate:
        ckpt_path = str(Path(__file__).parent / "checkpoints" / "ckpt_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = str(Path(__file__).parent / "checkpoints" / "ckpt_latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
        ctx = torch.tensor(tokenizer.encode(args.generate), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(ctx, max_new_tokens=500, temperature=args.temperature, top_k=args.top_k)
        print("\n── Generated ──")
        print(tokenizer.decode(out[0].tolist()))
        return

    # ── Train ──
    tcfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        resume=not args.no_resume,
        compile_model=args.compile,
    )

    trainer = Trainer(model, train_ds, val_ds, tokenizer, tcfg, device)
    trainer.train()

    # ── After training finishes, drop straight into chat ──
    print("\nTraining complete! Dropping into chat mode...")
    model.eval()
    chat_loop(model, tokenizer, device,
              temperature=args.temperature,
              top_k=args.top_k,
              max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
