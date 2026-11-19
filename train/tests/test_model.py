"""The Evaluate core: byte encoding with name marks, a tiny transformer, its training loop."""

import torch
from docta_orchestrator.search import Hit
from docta_train.model import CLS, Config, CoreEvaluate, Encoded, encode, pad_batch
from docta_train.training import train_steps

TINY = Config(d_model=32, layers=1, heads=2, ff=64, max_len=64)
S1 = Hit("s1", "Chronicle", "Varnelle was founded in 1432.", 0.9)


def test_encodes_question_then_excerpts_as_bytes():
    ids = encode("When?", [S1], max_len=512).ids
    assert ids[0] == CLS
    text = bytes(i - 3 for i in ids[1:]).decode()
    assert text == "Q: When?\n[Chronicle] Varnelle was founded in 1432.\n"


def test_marks_line_up_with_bytes_and_flag_the_question_names():
    e = encode("When was Varnelle founded?", [S1], max_len=512)
    assert len(e.marks) == len(e.ids) and e.marks[0] == 0  # CLS is no name
    text = bytes(i - 3 for i in e.ids[1:]).decode()
    flagged = "".join(c if m else " " for c, m in zip(text, e.marks[1:], strict=True))
    assert flagged.split() == ["Varnelle", "Varnelle"]


def test_truncates_bytes_and_marks_together_to_max_len():
    e = encode("x" * 1000, [S1], max_len=64)
    assert len(e.ids) == len(e.marks) == 64


def test_marks_change_the_prediction():
    torch.manual_seed(0)
    model = CoreEvaluate(TINY).eval()
    e = encode("Was Brask founded?", [], 64)
    unmarked = Encoded(e.ids, [0] * len(e.ids))
    with torch.no_grad():
        a, b = model(*pad_batch([e])), model(*pad_batch([unmarked]))
    assert not torch.allclose(a, b)


def test_padding_never_changes_a_prediction():
    torch.manual_seed(0)
    model = CoreEvaluate(TINY).eval()
    short, long_ = encode("Is Brask?", [], 64), encode("a much longer Brask question?", [S1], 64)
    with torch.no_grad():
        both, alone = model(*pad_batch([short, long_])), model(*pad_batch([short]))
    assert torch.allclose(both[0], alone[0], atol=1e-5)


def toy_batches():
    """Label = whether the text says 'yes': learnable in a few dozen steps."""
    texts = [("yes it is", 1.0), ("no it is not", 0.0)] * 32
    return [([encode(t, [], 64) for t, _ in texts], torch.tensor([y for _, y in texts]))]


def test_training_learns_a_toy_task():
    torch.manual_seed(0)
    model = CoreEvaluate(TINY)
    train_steps(model, toy_batches() * 60, lr=3e-3)
    batch = pad_batch([encode("yes it is", [], 64), encode("no it is not", [], 64)])
    with torch.no_grad():
        p = torch.sigmoid(model.eval()(*batch))
    assert p[0] > 0.9 and p[1] < 0.1


def test_training_is_deterministic():
    def trained():
        torch.manual_seed(0)
        model = CoreEvaluate(TINY)
        train_steps(model, toy_batches() * 3, lr=3e-3)
        return model.state_dict()

    a, b = trained(), trained()
    assert all(torch.equal(a[k], b[k]) for k in a)
