"""The Evaluate core: a tiny transformer over word tokens, and its training loop."""

import torch
from docta_train.model import Config, CoreEvaluate, pad_batch
from docta_train.training import train_steps

TINY = Config(d_model=32, layers=1, heads=2, ff=64, max_len=16)
VOCAB = 12


def test_pads_to_the_longest_sequence_and_masks_the_padding():
    ids, mask = pad_batch([[1, 5, 6], [1, 7]])
    assert ids.tolist() == [[1, 5, 6], [1, 7, 0]]
    assert mask.tolist() == [[False, False, False], [False, False, True]]


def test_padding_never_changes_a_prediction():
    torch.manual_seed(0)
    model = CoreEvaluate(TINY, VOCAB).eval()
    short, long_ = [1, 5, 6], [1, 7, 8, 9, 10, 11]
    with torch.no_grad():
        both, alone = model(*pad_batch([short, long_])), model(*pad_batch([short]))
    assert torch.allclose(both[0], alone[0], atol=1e-5)


def toy_batches():
    """Label = whether token 5 ('yes') is there: learnable in a few dozen steps."""
    seqs = [([1, 5, 9, 10], 1.0), ([1, 6, 9, 10, 11], 0.0)] * 32
    return [([s for s, _ in seqs], torch.tensor([y for _, y in seqs]))]


def test_training_learns_a_toy_task():
    torch.manual_seed(0)
    model = CoreEvaluate(TINY, VOCAB)
    train_steps(model, toy_batches() * 60, lr=3e-3)
    with torch.no_grad():
        p = torch.sigmoid(model.eval()(*pad_batch([[1, 5, 9, 10], [1, 6, 9, 10, 11]])))
    assert p[0] > 0.9 and p[1] < 0.1


def test_training_is_deterministic():
    def trained():
        torch.manual_seed(0)
        model = CoreEvaluate(TINY, VOCAB)
        train_steps(model, toy_batches() * 3, lr=3e-3)
        return model.state_dict()

    a, b = trained(), trained()
    assert all(torch.equal(a[k], b[k]) for k in a)
