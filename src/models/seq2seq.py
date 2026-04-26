"""Transformer encoder-decoder wrapper with beam search decoding."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_projection = nn.Linear(model_dim, vocab_size)
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_len = tgt.size(1)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.make_src_mask(src)
        memory = self.encoder(src, src_mask=src_mask)
        return memory, src_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = self.make_tgt_mask(tgt)
        decoder_output = self.decoder(tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.output_projection(decoder_output)

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src)
        return self.decode(tgt_input, memory=memory, src_mask=src_mask)

    def _blocked_tokens(self, sequence: list[int], no_repeat_ngram_size: int) -> set[int]:
        if no_repeat_ngram_size <= 1 or len(sequence) < no_repeat_ngram_size - 1:
            return set()

        prefix = tuple(sequence[-(no_repeat_ngram_size - 1):])
        observed: dict[tuple[int, ...], list[int]] = defaultdict(list)

        for idx in range(len(sequence) - no_repeat_ngram_size + 1):
            ngram = tuple(sequence[idx: idx + no_repeat_ngram_size])
            observed[ngram[:-1]].append(ngram[-1])

        return set(observed.get(prefix, []))

    def _beam_search_single(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int,
        max_len: int,
        min_len: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
    ) -> list[int]:
        beams = [(0.0, [self.bos_idx])]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            candidates: list[tuple[float, list[int]]] = []

            for score, sequence in beams:
                if sequence[-1] == self.eos_idx:
                    completed.append((score, sequence))
                    continue

                tgt = torch.tensor(sequence, dtype=torch.long, device=memory.device).unsqueeze(0)
                logits = self.decode(tgt=tgt, memory=memory, src_mask=src_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                # Block EOS until min_len tokens have been generated (excluding BOS)
                if len(sequence) < min_len:
                    log_probs[self.eos_idx] = torch.finfo(log_probs.dtype).min

                # Block repeated ngrams
                for token_id in self._blocked_tokens(sequence, no_repeat_ngram_size):
                    log_probs[token_id] = torch.finfo(log_probs.dtype).min

                top_scores, top_tokens = torch.topk(log_probs, beam_size)
                for candidate_score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                    candidates.append((score + candidate_score, sequence + [token]))

            if not candidates:
                break

            beams = sorted(
                candidates,
                key=lambda item: item[0] / ((len(item[1]) + 1) ** length_penalty),
                reverse=True,
            )[:beam_size]

        finalists = completed or beams
        best_score, best_sequence = max(
            finalists,
            key=lambda item: item[0] / ((len(item[1]) + 1) ** length_penalty),
        )
        del best_score
        return best_sequence

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        vocab=None,
        beam_size: int = 4,
        max_len: int = 128,
        min_len: int = 5,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ):
        self.eval()
        memory, src_mask = self.encode(src)
        predictions = []

        for batch_idx in range(src.size(0)):
            sequence = self._beam_search_single(
                memory=memory[batch_idx: batch_idx + 1],
                src_mask=src_mask[batch_idx: batch_idx + 1],
                beam_size=beam_size,
                max_len=max_len,
                min_len=min_len,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            predictions.append(sequence)

        if vocab is None:
            return predictions[0] if len(predictions) == 1 else predictions

        decoded = [vocab.decode(sequence) for sequence in predictions]
        return decoded[0] if len(decoded) == 1 else decoded
