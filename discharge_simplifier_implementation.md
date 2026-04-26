# Discharge Summary Simplifier — Implementation Guide
### NLP Course Project | Built From Scratch

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset & Preprocessing](#3-dataset--preprocessing)
4. [Core NLP Components (Built From Scratch)](#4-core-nlp-components-built-from-scratch)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation](#7-evaluation)
8. [Project Structure](#8-project-structure)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Project Overview

**Goal:** Build a system that takes a clinical discharge summary note and rewrites it in plain English a patient can understand — handling both summarization (condensing long notes) and paraphrasing (simplifying complex medical sentences).

**Dataset:** `medalpaca/medical_meadow_medical_flashcards` variant / MIMIC-III derived dataset  
- 158,114 (note, question, answer, task) pairs  
- Tasks include: Paraphrasing, Summarization, QA, etc.  
- **Filter to tasks:** `Paraphrasing` and `Summarization` for this project

**Constraints:**
- No end-to-end pretrained generation models (no BART, T5, GPT-2 fine-tuning, etc.)
- Word embeddings (GloVe, FastText, or similar) **are permitted**
- All model architecture must be implemented from scratch in PyTorch

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                           │
│  Raw Discharge Note → Sentence Splitter → Tokenizer        │
└────────────────────────────┬────────────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │   TEXT CLASSIFIER       │  ← built from scratch
                │  (Sentence complexity   │     BiLSTM
                │   + medical jargon      │
                │   detection)            │
                └────────────┬────────────┘
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐
    │ SIMPLE     │   │ PARAPHRASER  │   │ SUMMARIZER  │
    │ (pass-thru)│   │ (Transformer │   │ (Extractive │
    │            │   │  Enc-Dec)    │   │ + Abstractive│
    └─────┬──────┘   └───────┬──────┘   └──────┬──────┘
          └──────────────────┼──────────────────┘
                             │
                ┌────────────▼────────────┐
                │   POST-PROCESSOR        │
                │  (Fluency check,        │
                │   sentence reorder)     │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │   SIMPLIFIED OUTPUT     │
                └─────────────────────────┘
```

---

## 3. Dataset & Preprocessing

### 3.1 Filtering the Dataset

```python
from datasets import load_dataset

ds = load_dataset("medalpaca/medical_meadow_medical_flashcards")

# Filter to relevant tasks only
relevant_tasks = ['Paraphrasing', 'Summarization']
filtered = ds.filter(lambda x: x['task'] in relevant_tasks)

# Split: 80/10/10
split = filtered.train_test_split(test_size=0.2, seed=42)
val_test = split['test'].train_test_split(test_size=0.5, seed=42)
```

> **Tip:** You'll have roughly ~30–50k rows after filtering for Paraphrasing + Summarization tasks. That's plenty for training.

### 3.2 Vocabulary Building (From Scratch)

Build your own vocabulary — do **not** use a HuggingFace tokenizer.

```python
from collections import Counter
import re

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.freq = Counter()

    def tokenize(self, text):
        # Simple but effective medical-aware tokenizer
        text = text.lower()
        text = re.sub(r'(\d+\.?\d*)', r' \1 ', text)   # isolate numbers
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) # camelCase
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        return text.split()

    def build(self, texts):
        for text in texts:
            self.freq.update(self.tokenize(text))
        for word, count in self.freq.items():
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_len=512):
        tokens = self.tokenize(text)[:max_len]
        return [self.word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>

    def decode(self, indices):
        return ' '.join(self.idx2word.get(i, '<UNK>') for i in indices
                        if i not in (0, 2, 3))  # skip PAD, SOS, EOS

    def __len__(self):
        return len(self.word2idx)
```

### 3.3 GloVe Embedding Matrix

```python
import numpy as np
import torch

def load_glove(glove_path, vocab, embed_dim=100):
    """
    Load GloVe vectors and build an embedding matrix
    aligned to your custom vocabulary.
    """
    glove = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec

    emb_matrix = np.random.normal(0, 0.1, (len(vocab), embed_dim))
    emb_matrix[0] = 0  # PAD = zero vector

    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            emb_matrix[idx] = glove[word]
            hits += 1

    print(f"GloVe coverage: {hits}/{len(vocab)} ({100*hits/len(vocab):.1f}%)")
    return torch.FloatTensor(emb_matrix)
```

> Download: `glove.6B.100d.txt` from [nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)  
> You can also use **FastText** (`cc.en.300.bin`) for better subword coverage of rare medical terms.

### 3.4 Data Cleaning & Normalization

```python
import re

# Medical abbreviation expansion dictionary (extend as needed)
MEDICAL_ABBREVS = {
    'ards': 'acute respiratory distress syndrome',
    'hx': 'history',
    'dx': 'diagnosis',
    'tx': 'treatment',
    'rx': 'prescription',
    'pt': 'patient',
    'htn': 'hypertension',
    'dm': 'diabetes mellitus',
    'sob': 'shortness of breath',
    'cp': 'chest pain',
    'bid': 'twice daily',
    'tid': 'three times daily',
    'qid': 'four times daily',
    'prn': 'as needed',
    'po': 'by mouth',
    'iv': 'intravenous',
    'o2': 'oxygen',
    'bp': 'blood pressure',
    'hr': 'heart rate',
    'rr': 'respiratory rate',
}

def normalize_text(text):
    text = text.lower().strip()
    # Expand abbreviations
    for abbr, expansion in MEDICAL_ABBREVS.items():
        text = re.sub(rf'\b{abbr}\b', expansion, text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text
```

---

## 4. Core NLP Components (Built From Scratch)

### 4.1 Sentence Splitter

Split discharge notes into individual sentences for sentence-level classification.

```python
import re

def split_sentences(text):
    # Handle medical abbreviations that use periods (e.g., "Dr.", "Fig.")
    text = re.sub(r'(Dr|Mr|Mrs|Ms|Prof|Fig|vs|etc|No)\.\s', r'\1<PERIOD> ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences if s.strip()]
    return sentences
```

### 4.2 Sentence Complexity Classifier (BiLSTM — From Scratch)

Classify each sentence as **simple** (no rewriting needed) or **complex** (needs paraphrasing). This avoids over-generating on already-clear text.

```python
import torch
import torch.nn as nn

class SentenceComplexityClassifier(nn.Module):
    """
    Binary classifier: 0 = simple, 1 = complex/medical jargon.
    Architecture: Embedding → BiLSTM → Mean Pool → FC
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 pretrained_embeddings=None, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # fine-tune

        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                              bidirectional=True, num_layers=2,
                              dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Mean pooling over non-padded tokens
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(self.dropout(pooled))
```

> **Training signal:** You can create pseudo-labels using readability scores. Sentences with Flesch-Kincaid grade level > 10 → complex; ≤ 10 → simple.

```python
import textstat

def label_complexity(sentence):
    fk = textstat.flesch_kincaid_grade(sentence)
    return 1 if fk > 10 else 0
```

---

## 5. Model Architecture

### 5.1 Transformer Encoder-Decoder (Core Paraphraser)

This is the **heart** of the project. A full Transformer encoder-decoder built entirely from scratch in PyTorch — no `nn.Transformer`, no HuggingFace. Every sub-layer is hand-coded so the architecture is fully transparent and auditable for your course.

The overall data flow is:

```
src tokens → Embedding + Positional Encoding
           → N × TransformerEncoderLayer   (self-attention + FFN)
           → encoder memory [B, S, d_model]

tgt tokens → Embedding + Positional Encoding
           → N × TransformerDecoderLayer   (masked self-attn + cross-attn + FFN)
           → Linear projection → vocab logits
```

---

#### Positional Encoding

Transformers have no recurrence, so position information must be injected explicitly using sinusoidal encodings (Vaswani et al., 2017).

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal position signals to token embeddings.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)           # [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                            # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                         # [1, max_len, d_model]
        self.register_buffer('pe', pe)               # not a parameter

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

---

#### Multi-Head Scaled Dot-Product Attention

The core attention primitive, used in all three attention operations inside the Transformer (encoder self-attention, decoder masked self-attention, cross-attention).

```python
class MultiHeadAttention(nn.Module):
    """
    Splits d_model into h heads, computes scaled dot-product attention
    per head, then concatenates and projects back.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads   # dimension per head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: [B, T, d_model] → [B, h, T, d_k]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # x: [B, h, T, d_k] → [B, T, d_model]
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [B, T, d_model]
        mask: [B, 1, 1, T_k] (src padding) or [B, 1, T_q, T_k] (causal)
        """
        Q = self._split_heads(self.W_q(query))  # [B, h, T_q, d_k]
        K = self._split_heads(self.W_k(key))    # [B, h, T_k, d_k]
        V = self._split_heads(self.W_v(value))  # [B, h, T_k, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # [B, h, T_q, T_k]
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)       # [B, h, T_q, d_k]
        out = self.W_o(self._merge_heads(context))    # [B, T_q, d_model]
        return out, attn_weights
```

---

#### Position-wise Feed-Forward Network

Applied independently to each position after each attention sub-layer.

```python
class PositionwiseFFN(nn.Module):
    """
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    Inner dimension d_ff is typically 4 × d_model.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))
```

---

#### Encoder Layer & Encoder Stack

Each encoder layer applies self-attention → residual + LayerNorm → FFN → residual + LayerNorm.

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn       = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention sub-layer (Pre-LN variant for training stability)
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=512, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embeddings is not None:
            # GloVe dim may differ from d_model — project if needed
            glove_dim = pretrained_embeddings.size(1)
            if glove_dim != d_model:
                self.emb_proj = nn.Linear(glove_dim, d_model, bias=False)
                tmp_emb = nn.Embedding(vocab_size, glove_dim, padding_idx=0)
                tmp_emb.weight = nn.Parameter(pretrained_embeddings)
                self.embedding = tmp_emb
            else:
                self.emb_proj = nn.Identity()
                self.embedding.weight = nn.Parameter(pretrained_embeddings)
        else:
            self.emb_proj = nn.Identity()

        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers  = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.norm    = nn.LayerNorm(d_model)
        self.scale   = math.sqrt(d_model)

    def make_src_mask(self, src, pad_idx=0):
        # [B, 1, 1, S] — 0 where padded, 1 elsewhere
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src, pad_idx=0):
        mask = self.make_src_mask(src, pad_idx)
        x = self.emb_proj(self.embedding(src)) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x), mask   # memory + src_mask for decoder
```

---

#### Decoder Layer & Decoder Stack

Each decoder layer has three sub-layers: **masked** self-attention (causal), **cross-attention** over encoder memory, and FFN.

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn        = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        # 1) Masked self-attention (decoder tokens cannot attend to future)
        sa_out, _  = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        # 2) Cross-attention over encoder memory
        ca_out, cross_weights = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(ca_out))
        # 3) FFN
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x, cross_weights


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.norm      = nn.LayerNorm(d_model)
        self.fc_out    = nn.Linear(d_model, vocab_size)
        self.scale     = math.sqrt(d_model)

    def make_tgt_mask(self, tgt, pad_idx=0):
        """
        Combines padding mask and causal (look-ahead) mask.
        Shape: [B, 1, T_tgt, T_tgt] — 0 where masked.
        """
        T = tgt.size(1)
        pad_mask   = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        causal     = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        causal     = causal.unsqueeze(0).unsqueeze(0)             # [1,1,T,T]
        return pad_mask & causal

    def forward(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        x = self.embedding(tgt) * self.scale
        x = self.pos_enc(x)
        cross_attn_maps = []
        for layer in self.layers:
            x, cw = layer(x, memory, tgt_mask, src_mask)
            cross_attn_maps.append(cw)
        x = self.norm(x)
        return self.fc_out(x), cross_attn_maps  # [B, T, vocab], list of attn maps
```

---

#### Full Transformer Seq2Seq Wrapper

```python
class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx=0, tgt_pad_idx=0):
        super().__init__()
        self.encoder     = encoder
        self.decoder     = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src, tgt):
        """
        src: [B, S]   — source token ids (padded)
        tgt: [B, T]   — target token ids including <SOS>, excluding <EOS>
        Returns logits [B, T, vocab_size]
        """
        memory, src_mask = self.encoder(src, self.src_pad_idx)
        logits, _        = self.decoder(tgt, memory, src_mask)
        return logits

    @torch.no_grad()
    def generate(self, src, vocab, max_len=128, beam_size=4, device='cuda'):
        """
        Beam search decoding. Processes one example at a time (B=1).
        No recurrent state to carry — the decoder re-runs on the full
        generated prefix at each step (standard Transformer inference).
        """
        self.eval()
        sos_idx = vocab.word2idx['<SOS>']
        eos_idx = vocab.word2idx['<EOS>']

        memory, src_mask = self.encoder(src, self.src_pad_idx)

        # Each beam: (cumulative_log_prob, [token_ids])
        beams     = [(0.0, [sos_idx])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == eos_idx:
                    completed.append((score, seq))
                    continue
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                logits, _  = self.decoder(tgt_tensor, memory, src_mask)
                # Take logits at last position only
                log_probs  = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                top_scores, top_tokens = log_probs.topk(beam_size)
                for s, t in zip(top_scores.tolist(), top_tokens.tolist()):
                    candidates.append((score + s, seq + [t]))

            if not candidates:
                break

            # Length-normalised beam ranking
            beams = sorted(
                candidates,
                key=lambda x: x[0] / len(x[1]),
                reverse=True
            )[:beam_size]

            # Early stop if all beams ended
            if all(b[1][-1] == eos_idx for b in beams):
                completed.extend(beams)
                break

        if not completed:
            completed = beams

        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        return vocab.decode(best[1][1:])  # strip <SOS>
```

---

### 5.2 Extractive Summarizer (From Scratch)

For long notes, first **extract** the most important sentences, then **paraphrase** them. Use TF-IDF scoring (no sklearn — implement it).

```python
import math
from collections import Counter

class TFIDFExtractor:
    """
    Extractive summarizer using TF-IDF + sentence scoring.
    No sklearn. Pure Python + NumPy.
    """
    def __init__(self):
        self.idf = {}

    def _tokenize(self, text):
        return re.sub(r'[^a-z\s]', '', text.lower()).split()

    def fit(self, corpus):
        """corpus: list of documents (strings)"""
        N = len(corpus)
        df = Counter()
        for doc in corpus:
            words = set(self._tokenize(doc))
            df.update(words)
        self.idf = {w: math.log((N + 1) / (f + 1)) + 1
                    for w, f in df.items()}

    def _tf(self, tokens):
        count = Counter(tokens)
        total = len(tokens)
        return {w: c / total for w, c in count.items()}

    def score_sentences(self, sentences):
        scores = []
        for sent in sentences:
            tokens = self._tokenize(sent)
            tf = self._tf(tokens)
            score = sum(tf.get(w, 0) * self.idf.get(w, 0) for w in tokens)
            score = score / (len(tokens) + 1e-8)
            scores.append(score)
        return scores

    def summarize(self, text, ratio=0.4):
        sentences = split_sentences(text)
        if len(sentences) <= 2:
            return text
        scores = self.score_sentences(sentences)
        n_keep = max(1, int(len(sentences) * ratio))
        # Keep top-scoring sentences in their ORIGINAL ORDER
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_keep]
        )
        return ' '.join(sentences[i] for i in top_indices)
```

> **Extension:** Replace TF-IDF scores with sentence embeddings from your trained Transformer encoder (mean-pool the final encoder output over token positions) for **neural extractive summarization** — this is a strong experiment to include in your report, and it requires no extra training.

---

## 6. Training Pipeline

### 6.1 Dataset Class

```python
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class DischargeSummaryDataset(Dataset):
    def __init__(self, data, vocab, max_src=256, max_tgt=128):
        self.data    = data
        self.vocab   = vocab
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        note   = self.data[idx]['note']
        answer = self.data[idx]['answer']
        task   = self.data[idx]['task']

        if task == 'Summarization':
            src = self.vocab.encode(note, self.max_src)
        else:
            # Paraphrasing: extract the quoted sentence from the question
            question = self.data[idx]['question']
            match = re.search(r"'(.+?)'", question)
            src_text = match.group(1) if match else note
            src = self.vocab.encode(src_text, self.max_src)

        sos = self.vocab.word2idx['<SOS>']
        eos = self.vocab.word2idx['<EOS>']
        tgt = [sos] + self.vocab.encode(answer, self.max_tgt - 2) + [eos]

        return (torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))

def collate_fn(batch):
    """Pad src and tgt sequences to the longest in the batch."""
    srcs, tgts = zip(*batch)
    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)
    src_padded = torch.stack([F.pad(s, (0, max_src - s.size(0))) for s in srcs])
    tgt_padded = torch.stack([F.pad(t, (0, max_tgt - t.size(0))) for t in tgts])
    return src_padded, tgt_padded
```

### 6.2 Training Loop

The Transformer processes the entire target sequence in parallel during training (no step-by-step teacher forcing loop). The causal mask inside the decoder enforces left-to-right conditioning automatically.

```python
import torch.optim as optim

def train_epoch(model, dataloader, optimizer, criterion, clip=1.0, device='cuda'):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        # Decoder input: all tokens except last (<EOS>)
        # Decoder target: all tokens except first (<SOS>)
        tgt_in  = tgt[:, :-1]   # [B, T-1]
        tgt_out = tgt[:, 1:]    # [B, T-1]  ← what we predict

        logits = model(src, tgt_in)          # [B, T-1, vocab]
        logits = logits.reshape(-1, logits.size(-1))
        target = tgt_out.reshape(-1)

        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# Label smoothing loss — significantly helps Transformer generalization
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.pad_idx    = pad_idx
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        # pred: [N, V], target: [N]
        with torch.no_grad():
            smooth = torch.full_like(pred, self.smoothing / (self.vocab_size - 2))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth[:, self.pad_idx] = 0
            mask = (target == self.pad_idx)
            smooth[mask] = 0
        return F.kl_div(F.log_softmax(pred, dim=-1), smooth,
                        reduction='batchmean')
```

### 6.3 Hyperparameters & Training Config

```python
CONFIG = {
    # Transformer model dimensions
    'd_model':   256,     # embedding / hidden dimension throughout
    'n_heads':   8,       # attention heads (d_model must be divisible by n_heads)
    'd_ff':      1024,    # inner FFN dimension (typically 4 × d_model)
    'n_layers':  4,       # encoder and decoder stack depth
    'dropout':   0.1,

    # Word embeddings
    'glove_dim': 100,     # GloVe dimension; projected to d_model if different
    'max_src_len': 256,
    'max_tgt_len': 128,

    # Training
    'batch_size':     32,      # Transformers are memory-heavy; reduce if OOM
    'epochs':         30,
    'warmup_steps':   4000,    # for Noam scheduler (see 6.4)
    'clip':           1.0,
    'label_smoothing': 0.1,
    'min_word_freq':   2,

    # Decoding
    'beam_size':   4,
    'max_gen_len': 128,
}
```

> **On d_model vs GloVe dim:** GloVe-100 gives `embed_dim=100` but you likely want `d_model=256` for model capacity. The encoder projects GloVe embeddings linearly from 100 → 256 via `emb_proj`. Alternatively, use GloVe-300 and set `d_model=300` to skip the projection.

### 6.4 Learning Rate Scheduling — Noam (Warm-up + Inverse Square Root)

The original Transformer paper uses the **Noam schedule**: LR increases linearly for `warmup_steps` then decays proportionally to `1/sqrt(step)`. This is important — Transformers trained with a flat LR often diverge early.

```python
class NoamScheduler:
    """
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    Attach to any optimizer by calling .step() after each batch.
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self.factor        = factor
        self._step         = 0
        self._rate         = 0

    def step(self):
        self._step += 1
        rate = self._compute_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def _compute_lr(self):
        s, w = self._step, self.warmup_steps
        return self.factor * (self.d_model ** -0.5) * min(s ** -0.5, s * w ** -1.5)

    def zero_grad(self):
        self.optimizer.zero_grad()


# Instantiation — use Adam with β1=0.9, β2=0.98, ε=1e-9 (original paper values)
base_optimizer = torch.optim.Adam(
    model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
)
scheduler = NoamScheduler(base_optimizer, d_model=CONFIG['d_model'],
                          warmup_steps=CONFIG['warmup_steps'])
```

Then in your training loop, replace `optimizer.zero_grad()` / `optimizer.step()` with `scheduler.zero_grad()` / `scheduler.step()`.

---

## 7. Evaluation

### 7.1 ROUGE Score (From Scratch)

```python
from collections import Counter

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def rouge_n(hypothesis, reference, n=1):
    hyp_tokens = hypothesis.lower().split()
    ref_tokens  = reference.lower().split()
    hyp_ng = ngrams(hyp_tokens, n)
    ref_ng = ngrams(ref_tokens, n)
    overlap = sum((hyp_ng & ref_ng).values())
    precision = overlap / max(sum(hyp_ng.values()), 1)
    recall    = overlap / max(sum(ref_ng.values()), 1)
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    return {'precision': precision, 'recall': recall, 'f1': f1}

def rouge_l(hypothesis, reference):
    """Longest Common Subsequence based ROUGE-L"""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    m, n = len(hyp), len(ref)
    # LCS via DP
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / max(m, 1)
    recall    = lcs / max(n, 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### 7.2 Flesch-Kincaid Readability

```python
def count_syllables(word):
    word = word.lower()
    vowels = 'aeiouy'
    count  = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

def flesch_kincaid_grade(text):
    sentences = split_sentences(text)
    words = text.split()
    if not sentences or not words:
        return 0
    n_sentences = len(sentences)
    n_words     = len(words)
    n_syllables = sum(count_syllables(w) for w in words)
    asl = n_words / n_sentences          # avg sentence length
    asw = n_syllables / n_words          # avg syllables per word
    fk  = 0.39 * asl + 11.8 * asw - 15.59
    return round(fk, 2)

def flesch_reading_ease(text):
    sentences = split_sentences(text)
    words = text.split()
    if not sentences or not words:
        return 0
    n_syllables = sum(count_syllables(w) for w in words)
    asl = len(words)  / len(sentences)
    asw = n_syllables / len(words)
    return round(206.835 - 1.015 * asl - 84.6 * asw, 2)
```

### 7.3 BERTScore (Semantic Preservation)

This is the one place you **can** use a pretrained model — purely for evaluation, not for training.

```python
from bert_score import score as bert_score

def evaluate_semantic(predictions, references):
    P, R, F1 = bert_score(predictions, references,
                          lang='en', model_type='distilbert-base-uncased',
                          verbose=False)
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall':    R.mean().item(),
        'bertscore_f1':        F1.mean().item()
    }
```

### 7.4 Full Evaluation Suite

```python
def evaluate_model(model, dataloader, vocab, device):
    model.eval()
    all_preds, all_refs = [], []
    fk_before, fk_after = [], []

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            for i in range(src.size(0)):
                s = src[i:i+1]
                pred = model.generate(s, vocab, beam_size=4, device=device)
                ref  = vocab.decode(tgt[i].tolist())
                all_preds.append(pred)
                all_refs.append(ref)
                fk_before.append(flesch_kincaid_grade(vocab.decode(src[i].tolist())))
                fk_after.append(flesch_kincaid_grade(pred))

    r1  = np.mean([rouge_n(p, r, 1)['f1'] for p, r in zip(all_preds, all_refs)])
    r2  = np.mean([rouge_n(p, r, 2)['f1'] for p, r in zip(all_preds, all_refs)])
    rl  = np.mean([rouge_l(p, r)['f1']    for p, r in zip(all_preds, all_refs)])
    sem = evaluate_semantic(all_preds, all_refs)

    return {
        'ROUGE-1': round(r1, 4),
        'ROUGE-2': round(r2, 4),
        'ROUGE-L': round(rl, 4),
        'FK Grade (before)': round(np.mean(fk_before), 2),
        'FK Grade (after)':  round(np.mean(fk_after),  2),
        'FK Reduction':      round(np.mean(fk_before) - np.mean(fk_after), 2),
        **sem
    }
```

---

## 8. Project Structure

```
discharge_simplifier/
├── data/
│   ├── glove.6B.100d.txt          # GloVe embeddings (or glove.6B.300d.txt)
│   └── processed/
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
│
├── src/
│   ├── vocab.py                   # Vocabulary + medical tokenizer
│   ├── dataset.py                 # DischargeSummaryDataset + collate_fn
│   ├── embeddings.py              # GloVe loader + embedding matrix builder
│   ├── models/
│   │   ├── attention.py           # MultiHeadAttention (scaled dot-product)
│   │   ├── positional_encoding.py # Sinusoidal PositionalEncoding
│   │   ├── ffn.py                 # PositionwiseFFN
│   │   ├── encoder.py             # TransformerEncoderLayer + TransformerEncoder
│   │   ├── decoder.py             # TransformerDecoderLayer + TransformerDecoder
│   │   ├── transformer.py         # TransformerSeq2Seq wrapper + beam search
│   │   ├── complexity_classifier.py  # BiLSTM sentence classifier
│   │   └── tfidf_extractor.py    # TF-IDF extractive summarizer
│   ├── scheduler.py               # NoamScheduler (warm-up + inverse sqrt decay)
│   ├── loss.py                    # LabelSmoothingLoss
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # ROUGE + FK + BERTScore
│   ├── generate.py                # Inference pipeline (extract → simplify)
│   └── utils.py                   # Sentence splitter, normalization, misc
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_analysis.ipynb
│
├── checkpoints/
│   ├── best_transformer.pt
│   └── complexity_clf.pt
│
├── results/
│   └── evaluation_results.json
│
├── requirements.txt
└── README.md
```

---

## 9. Implementation Roadmap

### Phase 1 — Data & Vocab (Week 1)
- [ ] Load and filter dataset (keep Paraphrasing + Summarization tasks)
- [ ] Build custom `Vocabulary` class; tokenize all notes + answers
- [ ] Load GloVe → build embedding matrix; verify coverage over medical vocab
- [ ] Build `DischargeSummaryDataset` + `DataLoader` with `collate_fn`
- [ ] Build TF-IDF extractor and sanity-test on 5–10 sample notes

### Phase 2 — Complexity Classifier (Week 1–2)
- [ ] Auto-label sentences using Flesch-Kincaid grade threshold (> 10 = complex)
- [ ] Train `SentenceComplexityClassifier` (BiLSTM)
- [ ] Evaluate on held-out set; aim for > 80% accuracy

### Phase 3 — Transformer Core (Week 2–3)
- [ ] Implement `PositionalEncoding` and verify sinusoidal patterns visually
- [ ] Implement `MultiHeadAttention` — unit-test output shapes for various (B, T, d_model)
- [ ] Implement `PositionwiseFFN`, `TransformerEncoderLayer`, `TransformerDecoderLayer`
- [ ] Assemble `TransformerEncoder`, `TransformerDecoder`, `TransformerSeq2Seq`
- [ ] Wire in GloVe embedding matrix + linear projection if dims differ
- [ ] Implement `NoamScheduler` and `LabelSmoothingLoss`
- [ ] Run a smoke test: overfit on 100 examples to confirm the model can learn
- [ ] Train on full filtered dataset; monitor train/val loss curves
- [ ] Implement beam search decoding; check outputs qualitatively on 10 samples

### Phase 4 — Full Pipeline (Week 3)
- [ ] Connect TF-IDF extractor → Transformer paraphraser end-to-end
- [ ] For long notes: extract key sentences → simplify each with Transformer
- [ ] For short notes / single sentences: Transformer directly
- [ ] (Optional) Replace TF-IDF scoring with Transformer encoder embeddings for neural extraction

### Phase 5 — Evaluation & Report (Week 4)
- [ ] Run full eval suite: ROUGE-1/2/L, Flesch-Kincaid delta, BERTScore
- [ ] Error analysis: where does the model fail? (numbers, rare drug names, negation)
- [ ] Ablation studies (see Appendix: Baseline Comparisons)
- [ ] Write report

---

## Appendix: Baseline Comparisons

For your report, compare your model against at least two baselines:

| System | Description |
|---|---|
| **Copy baseline** | Output the source sentence unchanged |
| **Lead-N** | Take the first N sentences of the note |
| **TF-IDF only** | Extractive summary, no paraphrasing |
| **Transformer (greedy)** | Your model with greedy decoding |
| **Transformer (beam=4)** | Your model with beam search |
| **Ablation: no GloVe** | Random embedding init vs pretrained GloVe |
| **Ablation: n_layers=1** | Single-layer vs 4-layer encoder/decoder |
| **Ablation: n_heads=1** | Single-head vs multi-head attention |

The ablations are particularly compelling for an NLP course report — they let you empirically show *what each Transformer component contributes*.

---

## Appendix: Tips & Common Pitfalls

1. **Gradient instability early in training:** The Noam warm-up is not optional — without it, large gradient updates in the first few hundred steps will destabilize layer norms and lead to NaN loss. Always start with `lr=0` and let the scheduler ramp up.

2. **Padding mask shape bugs:** The most common Transformer implementation error. Encoder self-attention needs `[B, 1, 1, S]`, decoder cross-attention needs `[B, 1, 1, S]`, decoder causal mask needs `[B, 1, T, T]`. Getting these wrong causes silent incorrect attention — add shape assertions during debugging.

3. **Out-of-vocabulary drug names:** GloVe won't cover rare drug names like "remdesivir". Use FastText (`cc.en.300.bin`) instead — it constructs vectors for unseen words from character n-grams.

4. **Repetition in output:** Add a trigram repetition penalty in beam search: if a generated trigram has already appeared in the sequence, set its log-probability to −∞ before the `topk` selection.

5. **Slow beam search:** Transformer beam search reruns the full decoder on the growing prefix at each step — this is O(T²) per beam. For evaluation, use `batch_size=1` and keep `beam_size ≤ 5`. This is expected behaviour, not a bug.

6. **Vocab size vs d_model capacity:** Aim for ~20–30k vocabulary words. The output projection is `d_model × vocab_size` — a 256 × 30k matrix is ~30M parameters, which dominates the model. Tie encoder and decoder embeddings (`decoder.embedding.weight = encoder.embedding.weight`) to halve this and improve generalization.

7. **Embedding weight tying:** If you tie weights, make sure both encoder and decoder use the same `d_model` (no projection mismatch between embedding and model dim).

8. **Checkpointing:** Save the best model by validation ROUGE-L, not training loss. Transformer train loss can plateau while ROUGE continues improving for several epochs.
