"""Model factory helpers."""

from __future__ import annotations

import torch

from src.embeddings import build_embedding_wrapper
from src.models.complexity_classifier import SentenceComplexityClassifier
from src.models.decoder import TransformerDecoder
from src.models.encoder import TransformerEncoder
from src.models.seq2seq import TransformerSeq2Seq


def build_transformer_model(
    vocab_size: int,
    config: dict,
    pretrained_embeddings: torch.Tensor | None = None,
    embedding_wrapper=None,
    vocab=None,
):
    model_cfg = config["model"]
    if embedding_wrapper is None and vocab is not None:
        embedding_wrapper = build_embedding_wrapper(config, vocab=vocab)

    source_embed_dim = (
        int(getattr(embedding_wrapper, "embedding_dim", model_cfg["embed_dim"]))
        if embedding_wrapper is not None
        else model_cfg["embed_dim"]
    )
    target_embeddings = pretrained_embeddings
    if target_embeddings is None and embedding_wrapper is not None:
        apply_to_decoder = config.get("embeddings", {}).get("apply_to_decoder", True)
        if apply_to_decoder:
            target_embeddings = embedding_wrapper.static_matrix()

    target_embed_dim = int(target_embeddings.size(1)) if target_embeddings is not None else model_cfg["embed_dim"]
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=source_embed_dim,
        model_dim=model_cfg["model_dim"],
        ff_dim=model_cfg["ff_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_encoder_layers"],
        max_positions=model_cfg["max_positions"],
        dropout=model_cfg["dropout"],
        padding_idx=0,
        pretrained_embeddings=pretrained_embeddings,
        embedding_wrapper=embedding_wrapper,
    )
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=target_embed_dim,
        model_dim=model_cfg["model_dim"],
        ff_dim=model_cfg["ff_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_decoder_layers"],
        max_positions=model_cfg["max_positions"],
        dropout=model_cfg["dropout"],
        padding_idx=0,
        pretrained_embeddings=target_embeddings,
    )
    return TransformerSeq2Seq(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        pad_idx=0,
        bos_idx=2,
        eos_idx=3,
        model_dim=model_cfg["model_dim"],
    )


def build_complexity_classifier(
    vocab_size: int,
    config: dict,
    pretrained_embeddings: torch.Tensor | None = None,
):
    model_cfg = config["model"]
    return SentenceComplexityClassifier(
        vocab_size=vocab_size,
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["classifier_hidden_dim"],
        pretrained_embeddings=pretrained_embeddings,
        dropout=model_cfg["dropout"],
        padding_idx=0,
    )
