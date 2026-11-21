"""The Read core points at the answer; code copies the words found there."""

import torch
from docta_orchestrator.search import Hit
from docta_train.model import pad_batch
from docta_train.reader import (
    CoreRead,
    aligned,
    answer_positions,
    candidates,
    extract,
    pointer_loss,
)
from docta_train.words import BODY, SRC, STOP
from test_model import TINY, VOCAB
from test_words import vocab

S1 = Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432.", 0.9)
S2 = Hit("s2", "Notes", "Kel Dorn lives in Varnelle. Kel Dorn is 182 cm tall.", 0.9)
Q = "In what year was Varnelle founded?"
W = 9  # any word


def test_each_token_faces_its_word_and_the_source_whose_body_holds_it():
    words = aligned(Q, [S1])
    assert len(words) == len(vocab().tokens(Q, [S1]))
    assert words[:3] == [("<cls>", None), ("In", None), ("what", None)]
    assert ("Chronicle", None) in words  # titles are never pointed at
    assert words[-3:] == [("in", "s1"), ("1432", "s1"), (".", "s1")]


def test_the_answer_is_wherever_its_words_stand_in_its_sources():
    words = aligned("Where does Kel Dorn live?", [S1, S2])
    at = answer_positions(words, "Varnelle", ("s2",), max_len=256)
    assert [words[i] for i in at] == [("Varnelle", "s2")]  # not s1's Varnelle
    assert answer_positions(words, "Kel Dorn", ("s2",), 256) == [
        i for i, w in enumerate(words) if w == ("Kel", "s2")
    ]
    assert answer_positions(words, "1432", ("s1",), max_len=5) == []  # cut off


def test_code_copies_a_number_or_a_whole_name_with_its_source():
    words = aligned("Where does Kel Dorn live?", [S1, S2])
    assert extract(words, words.index(("1432", "s1"))) == ("1432", "s1")
    assert extract(words, words.index(("Kel", "s2"))) == ("Kel Dorn", "s2")


def test_only_body_words_are_candidates():
    ids, _ = pad_batch([[1, W, SRC, W, BODY, W, W, STOP, SRC, W, BODY, W]])
    assert candidates(ids).tolist() == [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]]


def test_the_core_scores_each_token_and_learns_to_point():
    torch.manual_seed(0)
    model = CoreRead(TINY, VOCAB)
    ids, pad = pad_batch([[1, W, SRC, W, BODY, 10, 11, STOP]] * 4)
    target = torch.zeros_like(ids, dtype=torch.bool)
    target[:, 6] = True
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(30):
        loss = pointer_loss(model(ids, pad), target, candidates(ids))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scores = model(ids, pad).masked_fill(~candidates(ids), float("-inf"))
    assert scores.argmax(1).tolist() == [6] * 4
