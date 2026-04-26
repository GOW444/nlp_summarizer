"""End-to-end simplification pipeline."""
from __future__ import annotations

import torch

from src.preprocessing import normalize_text, split_sentences


class SimplificationPipeline:
    def __init__(
        self,
        model,
        vocab,
        config: dict,
        device: torch.device,
        extractor=None,
        complexity_classifier=None,
        source_encoder=None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device
        self.extractor = extractor
        self.complexity_classifier = complexity_classifier
        self.source_encoder = source_encoder

    def _encode_sentence(self, sentence: str) -> torch.Tensor:
        max_src_len = self.config["data"]["max_src_len"]
        if self.source_encoder is not None:
            encoded = self.source_encoder.encode_source(sentence, max_len=max_src_len, vocab=self.vocab)
        else:
            encoded = self.vocab.encode(sentence, max_len=max_src_len)
        return torch.tensor(encoded, dtype=torch.long, device=self.device).unsqueeze(0)

    def _needs_simplification(self, sentence: str) -> bool:
        if self.complexity_classifier is None:
            return True
        encoded = self._encode_sentence(sentence)
        lengths = torch.tensor([encoded.size(1)], dtype=torch.long, device=self.device)
        logits = self.complexity_classifier(encoded, lengths)
        probability = torch.softmax(logits, dim=-1)[0, 1].item()
        return probability >= self.config["pipeline"]["complexity_threshold"]

    def simplify_sentence(self, sentence: str) -> str:
        encoded = self._encode_sentence(sentence)
        decoding_cfg = self.config["decoding"]
        return self.model.generate(
            src=encoded,
            vocab=self.vocab,
            beam_size=decoding_cfg["beam_size"],
            max_len=decoding_cfg["max_gen_len"],
            min_len=decoding_cfg.get("min_gen_len", 5),
            length_penalty=decoding_cfg["length_penalty"],
            no_repeat_ngram_size=decoding_cfg["no_repeat_ngram_size"],
        )

    def simplify(self, text: str) -> str:
        normalized = normalize_text(text)
        sentences = split_sentences(normalized)
        if self.extractor is not None and len(sentences) >= self.config["pipeline"]["long_note_sentence_threshold"]:
            working_text = self.extractor.summarize(
                normalized,
                ratio=self.config["data"]["summarization_ratio"],
            )
            sentences = split_sentences(working_text)
        outputs: list[str] = []
        for sentence in sentences:
            if not sentence:
                continue
            if self._needs_simplification(sentence):
                outputs.append(self.simplify_sentence(sentence))
            else:
                outputs.append(sentence)
        return " ".join(outputs).strip()
