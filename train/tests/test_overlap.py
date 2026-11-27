"""The overlap flag: which excerpt words the question also holds. The core
keyed its judgement on question templates and abstained on wordings it never
saw, even when an excerpt repeated the question word for word; the flag lets
it see that repetition whatever the template."""

import torch
from docta_train.fit_args import build_parser, config_of
from docta_train.model import Config, CoreEvaluate, overlap, pad_batch
from docta_train.words import BODY, SRC, STOP

CLS, GUILD, OF, NAME1, IS, OTHER = 1, 20, 21, 8, 22, 23


def test_excerpt_words_the_question_holds_are_flagged_and_nothing_else():
    # <cls> guild of <name_1> | <src> guild <body> <name_1> is other <stop>
    seq = [CLS, GUILD, OF, NAME1, SRC, GUILD, BODY, NAME1, IS, OTHER, STOP]
    ids, _ = pad_batch([seq, seq[:4] + [SRC, OTHER]])
    flags = overlap(ids)
    assert flags[0].tolist() == [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
    assert flags[1].tolist() == [0, 0, 0, 0, 0, 0] + [0] * 5  # padding never


def test_the_flag_changes_what_the_core_reads_only_when_turned_on():
    torch.manual_seed(0)
    seq = [CLS, GUILD, OF, SRC, GUILD, BODY, OTHER]
    ids, pad = pad_batch([seq])
    on = CoreEvaluate(Config(overlap=True, layers=1), 30)
    off = CoreEvaluate(Config(layers=1), 30)
    assert hasattr(on, "overlap") and not hasattr(off, "overlap")
    on.eval()
    with torch.no_grad():
        base = on(ids, pad)
        on.overlap.weight.zero_()
        assert not torch.equal(base, on(ids, pad))


def test_a_core_saved_before_the_flag_still_loads():
    old = {k: v for k, v in Config().__dict__.items() if k != "overlap"}
    assert Config(**old).overlap is False


def test_fit_turns_the_flag_on_by_option():
    assert config_of(build_parser().parse_args(["--overlap"])).overlap is True
    assert config_of(build_parser().parse_args([])).overlap is False
