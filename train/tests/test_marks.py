"""Name marks: which bytes repeat a name of the question, computed, not learned."""

from docta_train.marks import MAX_NAMES, name_marks, names


def marked(text, marks):
    """The marked runs of `text`, as (mark, substring)."""
    data, spans, i = text.encode(), [], 0
    while i < len(marks):
        j = i + 1
        while j < len(marks) and marks[j] == marks[i]:
            j += 1
        if marks[i]:
            spans.append((marks[i], data[i:j].decode()))
        i = j
    return spans


def test_names_are_the_capitalised_words_after_the_opening_one():
    assert names("In what year was the city where Orlan Teste lives founded?") == [
        "Orlan",
        "Teste",
    ]


def test_each_name_of_the_question_gets_its_own_mark_wherever_it_appears():
    question = "Who was born earlier, Orlan Teste or Brask?"
    text = f"Q: {question}\n[Notes] Brask met Orlan in Varnelle.\n"
    marks = name_marks(question, text)
    assert len(marks) == len(text.encode())
    assert marked(text, marks) == [
        (1, "Orlan"), (2, "Teste"), (3, "Brask"), (3, "Brask"), (1, "Orlan"),
    ]  # fmt: skip


def test_the_opening_question_word_is_not_a_name():
    marks = name_marks("Who is Brask?", "Q: Who is Brask?\nWho knows.\n")
    assert marked("Q: Who is Brask?\nWho knows.\n", marks) == [(1, "Brask")]


def test_only_whole_words_are_marked():
    text = "Brasker is not Brask."
    assert marked(text, name_marks("Where is Brask?", text)) == [(1, "Brask")]


def test_marks_follow_utf8_bytes():
    text = "Café de Brask"
    assert marked(text, name_marks("Where is Brask?", text)) == [(1, "Brask")]


def test_names_past_the_cap_share_the_last_mark():
    question = "Is Ana Bo Cel Dun Eve here?"
    assert MAX_NAMES == 4
    assert [m for m, _ in marked(question, name_marks(question, question))] == [1, 2, 3, 4, 4]


def test_leading_spaces_do_not_turn_the_opening_word_into_a_name():
    assert names("  Who is Brask?") == ["Brask"]
