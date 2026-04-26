# Architecture

This file mirrors the system described in `discharge_simplifier_implementation.md` and reflects the current implementation in `src/`.

## Component View

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         Raw Discharge Text                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src.preprocessing                                                  │
│ - normalize_text                                                    │
│ - split_sentences                                                   │
│ - prepare_source_text(note, question, task)                         │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src.dataset.DischargeSummaryDataset                                 │
│ - builds source text from note/question/task                         │
│ - encodes target answers with src.vocab.Vocabulary                   │
│ - delegates source encoding to EmbeddingWrapper when configured      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src.embeddings.EmbeddingWrapper                                      │
│                                                                      │
│ Static strategies:                                                   │
│   random, glove, gensim/word2vec/fasttext                            │
│   custom vocab ids -> nn.Embedding -> token vectors                  │
│                                                                      │
│ Contextual strategy:                                                 │
│   bert                                                               │
│   Hugging Face tokenizer ids -> AutoModel -> contextual token states │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ source feature dim is discovered here
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src.factory.build_transformer_model                                  │
│ - injects EmbeddingWrapper into TransformerEncoder                   │
│ - adjusts encoder input projection to wrapper.embedding_dim          │
│ - reuses static embedding matrices for the decoder when enabled      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ From-Scratch Transformer Seq2Seq                                     │
│                                                                      │
│ src.models.encoder.TransformerEncoder                                │
│   embedding/contextual features -> projection -> positional encoding │
│   -> encoder layers                                                  │
│                                                                      │
│ src.models.decoder.TransformerDecoder                                │
│   target vocab ids -> decoder layers -> output projection            │
│                                                                      │
│ src.models.seq2seq.TransformerSeq2Seq                                │
│   training forward pass + beam-search generation                     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Outputs                                                              │
│ - generated patient-friendly simplification                          │
│ - ROUGE/readability evaluation metrics                               │
│ - approach-specific checkpoints and training logs                    │
└──────────────────────────────────────────────────────────────────────┘
```

## Training And Comparison Flow

```text
python -m src.train
        │
        ├─ load config + CLI embedding overrides
        ├─ load processed train/val splits and custom Vocabulary
        ├─ build EmbeddingWrapper
        ├─ build Transformer model with dynamic source input dimension
        ├─ run tqdm training and validation loops
        ├─ append one row per epoch to:
        │    results/metrics_[approach_name].csv
        └─ save best checkpoints:
             checkpoints/best_model_[approach_name].pt
             checkpoints/best_model.pt
```

Each metrics CSV starts with an approach header such as:

```text
# EMBEDDING_TYPE=BERT APPROACH=bert_bert-base-uncased-finetune-false LR=2e-05 MODEL_DIM=256
```

The row format is comparison-friendly across approaches:

```text
epoch,embedding_type,approach_name,lr,train_loss,val_loss,score,epoch_time_sec,
trainable_params,is_best,ROUGE-1,ROUGE-2,ROUGE-L,FK Grade (before),
FK Grade (after),FK Reduction
```

## Embedding Strategies

| Strategy | Config value | Source preprocessing | Encoder input dim |
| --- | --- | --- | --- |
| Random trainable embeddings | `random` | custom `Vocabulary` | `embeddings.embed_dim` |
| GloVe | `glove` | custom `Vocabulary` | loaded GloVe dimension |
| Word2Vec/FastText via Gensim | `gensim`, `word2vec`, `fasttext` | custom `Vocabulary` | `KeyedVectors.vector_size` |
| BERT feature extractor/backbone | `bert` | Hugging Face tokenizer | `AutoModel.config.hidden_size` |

Static embeddings can initialize both the source encoder and decoder token embeddings. BERT is source-side because the decoder still generates tokens from the project vocabulary, preserving the from-scratch generation architecture described in the implementation guide.

## Inference Flow

```text
python inference.py --checkpoint checkpoints/best_model.pt --text "..."
        │
        ├─ load checkpoint config
        ├─ rebuild matching EmbeddingWrapper
        ├─ rebuild TransformerSeq2Seq
        ├─ preprocess raw text with the strategy-specific source encoder
        └─ return JSON:
             predicted_class, prediction, confidence, embedding_type, approach_name
```

For simplification checkpoints, `predicted_class` is reported as `simplified_text`; `prediction` contains the generated text and `confidence` is the mean greedy token probability.
