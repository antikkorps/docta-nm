"""Sentences: some attention heads read within one sentence only."""

from dataclasses import replace

import torch
from docta_orchestrator.search import Hit
from docta_train.model import CoreEvaluate, attention_mask, pad_batch, sentence_ids
from docta_train.words import BODY, SRC, STOP
from test_model import TINY, VOCAB
from test_words import vocab

W = 9  # any word


def test_a_full_stop_is_the_fixed_stop_token():
    hit = Hit("s1", "Chronicle of Brask", "Brask was founded in 1610. Brask was founded.", 0.5)
    v = vocab()
    tokens = v.tokens("In what year was Brask founded?", [hit])
    assert tokens.count("<stop>") == 2 and "." not in tokens
    assert "." not in v.words
    assert v.index["<stop>"] == STOP


def test_question_titles_and_body_sentences_are_numbered_in_order():
    ids = torch.tensor([[1, W, W, SRC, W, BODY, W, STOP, W, W, STOP, SRC, W, BODY, W, STOP]])
    assert sentence_ids(ids).tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5]]


def test_padding_is_in_no_sentence():
    ids, _ = pad_batch([[1, W, SRC, BODY, W, STOP], [1, W]])
    assert sentence_ids(ids)[1].tolist() == [0, 0, -1, -1, -1, -1]


def test_local_heads_see_their_sentence_and_the_others_everything_but_padding():
    ids, _ = pad_batch([[1, W, SRC, W], [1, W]])
    blocked = attention_mask(ids, heads=2, local=1).view(2, 2, 4, 4)
    assert (
        blocked[0, 0].tolist()
        == [[False, False, True, True]] * 2 + [[True, True, False, False]] * 2
    )
    assert not blocked[0, 1].any()
    # Padding sits in no sentence but sees itself: a row with nothing to see
    # would give NaN.
    assert blocked[1, 0, 2:].tolist() == [[True, True, False, False]] * 2


def test_with_local_heads_only_the_question_never_sees_the_excerpts():
    torch.manual_seed(0)
    local = CoreEvaluate(replace(TINY, local_heads=TINY.heads), VOCAB).eval()
    shared = CoreEvaluate(replace(TINY, local_heads=0), VOCAB).eval()
    a, b = (
        pad_batch([[1, W, SRC, 10, BODY, 10, STOP]]),
        pad_batch([[1, W, SRC, 11, BODY, 11, STOP]]),
    )
    assert local(*a).item() == local(*b).item()
    assert shared(*a).item() != shared(*b).item()


def test_padding_changes_nothing_and_gives_no_nan():
    torch.manual_seed(0)
    model = CoreEvaluate(TINY, VOCAB).eval()
    seq = [1, W, SRC, 10, BODY, 10, STOP]
    alone = model(*pad_batch([seq]))
    padded = model(*pad_batch([seq, seq + [W, W, STOP, W]]))
    assert not padded.isnan().any()
    assert torch.allclose(alone, padded[:1], atol=1e-5)
