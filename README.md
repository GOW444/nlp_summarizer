# Discharge Summary Simplifier

From-scratch NLP course project for simplifying clinical discharge summaries into patient-friendly English.

The original implementation guide centered the generation module around an LSTM encoder-decoder with Bahdanau attention. This scaffold intentionally replaces that core with a from-scratch Transformer encoder-decoder while keeping the rest of the project aligned with the guide:

- custom vocabulary and tokenizer
- custom dataset + collate pipeline
- optional GloVe embeddings
- BiLSTM sentence complexity classifier
- TF-IDF extractive summarizer
- from-scratch Transformer encoder-decoder for paraphrasing and abstractive simplification
- ROUGE, readability, and optional BERTScore evaluation

## Current Status

The full project structure is in place, but no dataset artifacts are checked in yet. The code is designed to fail clearly when `data/processed/train.pt`, `val.pt`, `test.pt`, and `vocab.json` are missing.

## Project Layout

```text
.
├── discharge_simplifier_implementation.md
├── data/
│   └── processed/
├── checkpoints/
├── notebooks/
├── results/
├── requirements.txt
└── src/
    ├── config.py
    ├── dataset.py
    ├── embeddings.py
    ├── evaluate.py
    ├── factory.py
    ├── generate.py
    ├── pipeline.py
    ├── prepare_data.py
    ├── preprocessing.py
    ├── train.py
    ├── utils.py
    ├── vocab.py
    └── models/
        ├── attention.py
        ├── complexity_classifier.py
        ├── decoder.py
        ├── encoder.py
        ├── seq2seq.py
        └── tfidf_extractor.py
```

## Planned Workflow

1. Prepare dataset artifacts once we add or download the dataset:

```bash
python -m src.prepare_data
```

2. Train the Transformer simplifier:

```bash
python -m src.train
```

3. Run evaluation:

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

4. Generate simplifications:

```bash
python -m src.generate --checkpoint checkpoints/best_model.pt --text "Patient with hx of htn admitted for sob..."
```

## Data Expectations

The prep script expects the `medalpaca/medical_meadow_medical_flashcards` dataset fields described in the implementation guide:

- `note`
- `question`
- `answer`
- `task`

It filters to:

- `Paraphrasing`
- `Summarization`

and writes:

- `data/processed/train.pt`
- `data/processed/val.pt`
- `data/processed/test.pt`
- `data/processed/vocab.json`

## Notes

- No pretrained generator is used.
- Word embeddings such as GloVe are supported through `data/glove.6B.100d.txt`.
- BERTScore is optional and only used for evaluation.
