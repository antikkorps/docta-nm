"""Which epoch Evaluate keeps: the one that answers right most often on val."""

from docta_train.fit import better


def epoch(accuracy, loss):
    return {"val_accuracy": accuracy, "val_loss": loss}


def test_a_more_accurate_epoch_wins_even_with_a_higher_loss():
    # Palier 1, seed 2: epoch 1 had the lowest loss (0.70) at 55.2%, epoch 6
    # read 71.3% at 0.87, an overconfident core's loss rising as it gets more right.
    assert better(epoch(0.713, 0.865), epoch(0.552, 0.700))


def test_a_less_accurate_epoch_never_wins():
    assert not better(epoch(0.60, 0.50), epoch(0.65, 0.90))


def test_equal_accuracy_goes_to_the_lower_loss():
    assert better(epoch(0.65, 0.60), epoch(0.65, 0.70))
    assert not better(epoch(0.65, 0.70), epoch(0.65, 0.60))


def test_the_first_epoch_is_kept():
    assert better(epoch(0.5, 0.7), None)
