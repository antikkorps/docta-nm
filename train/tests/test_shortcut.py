"""Question shape: the label must not be guessable without reading."""

from docta_train.evaluate_data import Example
from docta_train.shortcut import balance, names, shape, shape_accuracy

CHAIN = "In what year was the city where Orlan Teste lives founded?"
RETRIEVE = "In what year was Varnelle founded?"


def example(i, question, label):
    return Example("w", f"q{i}", "train", question, (), label)


def test_names_are_the_capitalised_words():
    assert names(CHAIN) == ["In", "Orlan", "Teste"]


def test_shape_hides_names_and_numbers():
    assert shape(CHAIN) == "_ what year was the city where _ _ lives founded?"
    assert shape("Was Brask founded before 1432?") == "_ _ founded before #?"


def test_balance_keeps_as_many_enough_as_not_enough_per_shape():
    data = [example(i, CHAIN, i < 2) for i in range(6)]  # 2 enough, 4 not
    data += [example(10 + i, RETRIEVE, True) for i in range(3)]  # one label only
    kept = balance(data, seed=0)
    assert sorted(e.label for e in kept) == [False, False, True, True]
    assert all(e.question == CHAIN for e in kept)


def test_balance_is_deterministic_and_keeps_the_input_order():
    data = [example(i, CHAIN, i % 3 == 0) for i in range(30)]
    a, b = balance(data, seed=7), balance(data, seed=7)
    assert a == b
    ids = [int(e.question_id[1:]) for e in a]
    assert ids == sorted(ids)


def test_shape_accuracy_is_what_guessing_from_the_shape_alone_scores():
    train = [example(0, CHAIN, False), example(1, CHAIN, False), example(2, RETRIEVE, True)]
    val = [example(3, CHAIN, False), example(4, CHAIN, True), example(5, RETRIEVE, True)]
    assert shape_accuracy(train, val) == 2 / 3


def test_shape_accuracy_falls_back_to_the_overall_majority_on_unseen_shapes():
    train = [example(0, CHAIN, True), example(1, CHAIN, True), example(2, RETRIEVE, False)]
    val = [example(3, "Who is taller, Brask or Varnelle?", True)]
    assert shape_accuracy(train, val) == 1.0
