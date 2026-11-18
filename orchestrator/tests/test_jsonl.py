import json

import pytest
from docta_orchestrator.jsonl import WorldBatch, group_worlds, read_records
from orch_sample import QUESTIONS, RECORDS, SOURCES, WORLD


def write_lines(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_reads_every_record_in_order_with_its_location(tmp_path):
    path = write_lines(tmp_path / "a.jsonl", RECORDS)
    located = list(read_records(path))
    assert [record for _, record in located] == RECORDS
    assert located[2][0] == f"{path}:3"


def test_rejects_another_schema_version_with_its_line(tmp_path):
    path = write_lines(tmp_path / "a.jsonl", [WORLD, {**SOURCES[0], "schema": "v1"}])
    with pytest.raises(ValueError, match=r"a\.jsonl:2.*v1"):
        list(read_records(path))


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]"])
def test_rejects_a_line_that_is_not_a_json_object(tmp_path, bad):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(WORLD) + "\n" + bad + "\n")
    with pytest.raises(ValueError, match=r"a\.jsonl:2"):
        list(read_records(path))


def test_rejects_a_missing_field_by_name(tmp_path):
    question = {k: v for k, v in QUESTIONS[0].items() if k != "required_sources"}
    path = write_lines(tmp_path / "a.jsonl", [WORLD, question])
    with pytest.raises(ValueError, match=r"a\.jsonl:2.*required_sources"):
        list(read_records(path))


def test_rejects_a_missing_field_inside_a_world(tmp_path):
    world = {**WORLD, "facts": [{"id": "f1", "subj": "e1", "pred": "founded_in"}]}
    path = write_lines(tmp_path / "a.jsonl", [world])
    with pytest.raises(ValueError, match=r"a\.jsonl:1.*obj"):
        list(read_records(path))


def test_rejects_an_unknown_record_type(tmp_path):
    path = write_lines(tmp_path / "a.jsonl", [WORLD, {**SOURCES[0], "type": "note"}])
    with pytest.raises(ValueError, match=r"a\.jsonl:2.*unknown record type 'note'"):
        list(read_records(path))


def located(records):
    return [(f"x:{i}", r) for i, r in enumerate(records, start=1)]


def test_groups_sources_and_questions_under_their_world():
    second = {**WORLD, "world_id": "w_test02"}
    batches = list(group_worlds(located([*RECORDS, second])))
    assert batches == [WorldBatch(WORLD, SOURCES, QUESTIONS), WorldBatch(second, [], [])]


@pytest.mark.parametrize(
    ("records", "where"),
    [([SOURCES[0]], "x:1"), ([WORLD, {**QUESTIONS[0], "world_id": "w_other"}], "x:2")],
)
def test_rejects_a_record_outside_its_world(records, where):
    with pytest.raises(ValueError, match=f"{where}: .*before its world"):
        list(group_worlds(located(records)))
