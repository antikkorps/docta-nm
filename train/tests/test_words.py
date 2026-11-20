"""Word tokens: the templates' vocabulary from train, names as their rank in the question."""

from docta_orchestrator.search import Hit
from docta_train.evaluate_data import Example
from docta_train.words import MAX_NAMES, Vocab, names

S1 = Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432.", 0.9)


def example(world, question, *hits):
    return Example(world, "q", "train", question, hits, True)


def vocab(min_worlds=2):
    """Template words in two worlds; each name in one world only."""
    return Vocab.build(
        [
            example("w1", "In what year was Varnelle founded?", S1),
            example("w2", "In what year was Brask founded?",
                    Hit("s2", "Chronicle of Brask", "Brask was founded in 1610.", 0.5)),
        ],
        min_worlds,
    )  # fmt: skip


def test_names_are_the_capitalised_words_after_the_opening_one():
    question = "  In what year was the city where Orlan Teste lives founded?"
    assert names(question) == ["Orlan", "Teste"]


def test_vocabulary_keeps_words_seen_in_enough_worlds_lowercased():
    words = vocab().words
    assert {"in", "what", "chronicle", "founded", "?", "."} <= set(words)
    assert "varnelle" not in words and "brask" not in words


def test_question_then_each_excerpt_as_word_tokens():
    v = vocab()
    tokens = v.tokens("In what year was Varnelle founded?", [S1])
    assert tokens == [
        "<cls>", "in", "what", "year", "was", "<name_1>", "founded", "?",
        "<src>", "chronicle", "of", "<name_1>",
        "<body>", "<name_1>", "was", "founded", "in", "<num>", ".",
    ]  # fmt: skip
    assert v.encode("In what year was Varnelle founded?", [S1]) == [v.index[t] for t in tokens]


def test_other_names_and_unseen_words_get_their_own_tokens():
    hit = Hit("s3", "Notes", "Brask trades amber with Varnelle.", 0.5)
    assert vocab().tokens("Where is Brask?", [hit])[-7:] == [
        "<body>", "<name_1>", "<unk>", "<unk>", "<unk>", "<name>", ".",
    ]  # fmt: skip


def test_names_past_the_cap_share_the_last_rank():
    tokens = vocab().tokens("Is Ana Bo Cel Dun Eve here?", [])
    assert MAX_NAMES == 4
    assert [t for t in tokens if t.startswith("<name_")] == [f"<name_{k}>" for k in (1, 2, 3, 4, 4)]


def test_vocabulary_is_deterministic_and_round_trips():
    assert vocab().words == vocab().words
    assert Vocab(vocab().words).index == vocab().index


def test_question_words_count_even_without_excerpts():
    seen = Vocab.build([example("w1", "Who knows?"), example("w2", "Who knows?")], 2).words
    assert {"who", "knows"} <= set(seen)
