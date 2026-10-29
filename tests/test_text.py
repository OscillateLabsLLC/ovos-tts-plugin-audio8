from ovos_tts_plugin_audio8 import text


ELVIS = (
    'Elvis Presley Enterprises, Inc. is a corporate entity created by "The Elvis Presley Trust" to conduct '
    "business and manage its assets, including Graceland. EPE's business extends far beyond the Graceland "
    "operation, and includes worldwide licensing of Elvis-related products and ventures, the development of "
    "Elvis-related music, film, video, television and stage productions, the ongoing development of EPE's "
    "Internet presence and the management of significant music publishing assets."
)


def test_short_sentences_stay_whole():
    line = "Hailing frequencies open. The Romulan commander wishes to speak with you."
    assert text.split_for_streaming(line) == text.split_sentences(line)


def test_long_sentences_split_into_bounded_pieces_without_stubs():
    pieces = text.split_for_streaming(ELVIS)
    assert all(len(piece) <= text.MAX_STREAM_CHARS for piece in pieces), pieces
    assert " ".join(pieces).split() == ELVIS.split()
    for before, after in zip(pieces, pieces[1:]):
        if before[-1] not in ",;:.!?—":  # a word cut, only inside a clause over the limit: both halves balanced
            assert min(len(before), len(after)) >= text.MIN_STREAM_PIECE_CHARS, (before, after)


def test_no_tiny_leading_clause():
    line = ("Captain, the Romulan warbird has decloaked off our port bow, and its disruptor banks are fully "
            "charged and locked onto our primary hull.")
    pieces = text.split_for_streaming(line)
    assert not pieces[0].endswith("Captain,")
    assert all(len(piece) <= text.MAX_STREAM_CHARS for piece in pieces)


def test_a_clause_without_punctuation_splits_at_words():
    line = " ".join(["word"] * 60)
    pieces = text.split_for_streaming(line)
    assert all(text.MIN_STREAM_PIECE_CHARS <= len(piece) <= text.MAX_STREAM_CHARS for piece in pieces), pieces
    assert " ".join(pieces) == line
