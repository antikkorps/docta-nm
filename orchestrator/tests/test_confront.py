"""Hand-written Confront: the naive vote, and the one that counts copies once."""

from docta_orchestrator.loop import run
from docta_orchestrator.models.confront import ChallengeModel, MajorityModel, claims, origin_of
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit

QUESTION = "In what year was Varnelle founded?"
TRUTH = Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432.", 0.9)
ORIGIN = Hit("s2", "Notes on Varnelle", "The founding of Varnelle dates back to 1450.", 0.62)
COPIES = [
    Hit("s3", "Register of Varnelle",
        "According to Notes on Varnelle, the founding of Varnelle dates back to 1450.", 0.8),
    Hit("s4", "Annals of Varnelle",
        "As Notes on Varnelle reports, the founding of Varnelle dates back to 1450.", 0.75),
]  # fmt: skip
OTHER = Hit("s5", "Records of Varnelle", "Varnelle has a population of 5000.", 0.7)


def ask(model, hits, question=QUESTION):
    return run(question, model, lambda q: hits)[0]


def test_reads_who_says_what_and_whom_a_copy_credits():
    found = claims(QUESTION, (TRUTH, ORIGIN, *COPIES, OTHER))
    assert [(c.value, c.source_id, c.credits) for c in found] == [
        ("1432", "s1", None),
        ("1450", "s2", None),
        ("1450", "s3", "Notes on Varnelle"),
        ("1450", "s4", "Notes on Varnelle"),
    ]


def test_the_naive_vote_falls_for_a_false_consensus():
    outcome = ask(MajorityModel(), [TRUTH, ORIGIN, *COPIES])
    assert outcome.answer == "1450"


def test_challenge_counts_copies_once_and_follows_the_reliable_source():
    outcome = ask(ChallengeModel(), [TRUTH, ORIGIN, *COPIES])
    assert (outcome.answer, outcome.cited, outcome.exit) == ("1432", ("s1",), None)


def test_challenge_spots_copies_whose_origin_was_not_read():
    outcome = ask(ChallengeModel(), [TRUTH, *COPIES])
    assert outcome.answer == "1432"  # two copies of one origin weigh as one


def test_challenge_refuses_to_settle_two_equally_reliable_sources():
    rival = Hit("s2", "Notes on Varnelle", "Varnelle has existed since 1450.", 0.87)
    outcome = ask(ChallengeModel(), [TRUTH, rival])
    assert (outcome.answer, outcome.exit) == (None, Exit.CONTRADICTION)
    assert ask(MajorityModel(), [TRUTH, rival]).exit is None  # the vote always settles


# Only the credit's wording is unseen here. In the ood split the sentences
# are held out too, so these rules cannot read them and fall back to keyword.
def test_a_credit_in_an_unseen_wording_still_marks_a_copy():
    held_out = Hit("s3", "Register of Varnelle",
                   "Notes on Varnelle records that the founding of Varnelle dates back to 1450.",
                   0.8)  # fmt: skip
    found = claims(QUESTION, (held_out,))
    assert [(c.value, c.credits) for c in found] == [("1450", "Notes on Varnelle")]


def test_a_question_without_disagreement_is_answered_as_before():
    for model in (MajorityModel(), ChallengeModel()):
        outcome = ask(model, [TRUTH, OTHER])
        assert (outcome.answer, outcome.cited) == ("1432", ("s1",))


def test_a_credit_is_never_read_as_part_of_the_object():
    # "{o} is home to {s}." would swallow the credit into the object.
    copy = Hit("s3", "Register of Kel Dorn",
               "As Notes on Kel Dorn reports, Brask is home to Kel Dorn.", 0.8)  # fmt: skip
    found = claims("In which city does Kel Dorn live?", (copy,))
    assert [(c.value, c.credits) for c in found] == [("Brask", "Notes on Kel Dorn")]


def test_too_close_to_call_is_inclusive_at_five_hundredths():
    at = Hit("s2", "Notes on Varnelle", "Varnelle has existed since 1450.", 0.85)
    assert ask(ChallengeModel(), [TRUTH, at]).exit == Exit.CONTRADICTION  # 0.90 vs 0.85
    above = Hit("s2", "Notes on Varnelle", "Varnelle has existed since 1450.", 0.84)
    assert ask(ChallengeModel(), [TRUTH, above]).answer == "1432"  # 0.06 apart


def test_challenge_does_not_conclude_on_copies_alone():
    outcome = ask(ChallengeModel(), COPIES)
    assert (outcome.answer, outcome.exit) == (None, Exit.NOT_FOUND)


def test_one_source_one_vote_and_reliability_breaks_ties():
    twice = Hit(
        "s2",
        "Notes on Varnelle",
        "Varnelle has existed since 1450. Varnelle has existed since 1450.",
        0.85,
    )
    # One vote each, then 0.90 beats 0.85; counting sentences would say 1450.
    assert ask(MajorityModel(), [TRUTH, twice]).answer == "1432"


def test_two_true_sources_sharing_a_title_stay_independent():
    twin = Hit("s6", "Chronicle of Varnelle", "The founding of Varnelle dates back to 1432.", 0.7)
    rival = Hit("s2", "Notes on Varnelle", "Varnelle has existed since 1450.", 0.92)
    outcome = ask(ChallengeModel(), [TRUTH, twin, rival])
    assert outcome.answer == "1432"  # 0.90 + 0.70 against 0.92


def test_a_credit_to_a_title_shared_by_two_read_sources_names_the_last_one():
    # Pins the last-wins title map in origin_of: origin titles are unique by
    # construction, so this only happens if that guarantee is ever broken.
    first = Hit("s2", "Notes on Varnelle", "Varnelle has existed since 1450.", 0.62)
    second = Hit("s7", "Notes on Varnelle", "Varnelle has existed since 1450.", 0.61)
    found = claims(QUESTION, (first, second, COPIES[0]))
    copy = next(c for c in found if c.source_id == "s3")
    assert origin_of(copy, found) == "s7"
