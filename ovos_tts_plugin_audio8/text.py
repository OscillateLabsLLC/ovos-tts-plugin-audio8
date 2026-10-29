"""How utterances are cut into generation units."""

import re
from typing import List

import quebra_frases


# Streaming re-decodes each unit from its first frame, so the decode cost per chunk grows with the unit.
# A ~50-word Wikipedia sentence (~390 frames) costs ~1.6 s per decode on MPS, more than a chunk's audio,
# and stutters whatever the schedule. Units up to ~100 characters (~5 s) streamed without gaps.
MAX_STREAM_CHARS = 100
MIN_STREAM_PIECE_CHARS = 40
_CLAUSE_BREAK = re.compile(r"(?<=[,;:—])\s+")


def split_sentences(utterance: str) -> List[str]:
    return [part for part in quebra_frases.sentence_tokenize(utterance) if part.strip()] or [utterance]


def split_for_streaming(utterance: str) -> List[str]:
    """Sentences, with long ones cut at clause punctuation (or word boundaries) into 40-100 character pieces."""
    return [piece for sentence in split_sentences(utterance) for piece in _split_long(sentence)]


def _split_long(sentence: str) -> List[str]:
    """Clause pieces, merging a short one into its neighbour only while the result still fits.

    A short clause beats a cut mid-phrase ("... manage its" / "assets, including Graceland.").
    """
    if len(sentence) <= MAX_STREAM_CHARS:
        return [sentence]
    pieces: List[str] = []
    for clause in _CLAUSE_BREAK.split(sentence):
        short = pieces and min(len(pieces[-1]), len(clause)) < MIN_STREAM_PIECE_CHARS
        if short and len(pieces[-1]) + 1 + len(clause) <= MAX_STREAM_CHARS:
            pieces[-1] = f"{pieces[-1]} {clause}"
        else:
            pieces.append(clause)
    return [chunk for piece in pieces for chunk in _split_words(piece)]


def _split_words(text: str) -> List[str]:
    """Last resort for one clause over the limit: cut into equal parts at the nearest word boundaries."""
    if len(text) <= MAX_STREAM_CHARS:
        return [text]
    words, parts = text.split(" "), -(-len(text) // MAX_STREAM_CHARS)
    target, chunks, current = len(text) / parts, [], []
    for word in words:
        if current and len(" ".join(current)) + len(word) / 2 > target and len(chunks) < parts - 1:
            chunks.append(" ".join(current))
            current = []
        current.append(word)
    return chunks + [" ".join(current)]
