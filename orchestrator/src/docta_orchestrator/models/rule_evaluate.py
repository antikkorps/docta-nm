"""RuleEvaluateModel: keyword's plan and conclusion, a hand-written Evaluate.

The rule knows the seen templates: it reads which facts the question needs and
checks each one is stated in an excerpt. On seen splits it parses the
generator's own templates, so it is close to an oracle there: a trained core
can hardly beat it on test. Only ood, whose phrasings are new, can show a core
learned more than the templates. On a phrasing it does not know, the rule
falls back to keyword's judgement.
"""

import re
from functools import cache

from docta_orchestrator.models.keyword import KeywordModel
from docta_orchestrator.models.templates import CHAIN_LINKS, COMPARISONS, QUESTIONS, SENTENCES
from docta_orchestrator.states import Evidence, Exit, Framing, Verdict

Statement = tuple[str, str, str]  # (subject, predicate, object), as written
Need = tuple[str, tuple[str, ...]]  # a subject, then the predicates to follow from it


@cache
def pattern(template: str) -> re.Pattern[str]:
    escaped = re.escape(template).replace(r"\{s\}", "(?P<s>.+?)").replace(r"\{o\}", "(?P<o>.+?)")
    return re.compile(escaped)


NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"


@cache
def sentence_pattern(template: str) -> re.Pattern[str]:
    """A sentence states a name and a name or number, as worldgen fills it: a
    looser group would let "{o}" swallow a frame ("Records show that 1432")."""
    escaped = re.escape(template).replace(r"\{s\}", f"(?P<s>{NAME})")
    return re.compile(escaped.replace(r"\{o\}", rf"(?P<o>{NAME}|\d+)"))


def read_sentence(sentence: str) -> list[Statement]:
    """Every fact a seen sentence template states; none for anything else."""
    found = []
    for pred, templates in SENTENCES.items():
        for template in templates:
            m = sentence_pattern(template).fullmatch(sentence)
            if m:
                found.append((m["s"], pred, m["o"]))
    return found


def needs(question: str) -> list[Need] | None:
    """The facts a seen question needs; None when its phrasing is unknown."""
    for pred, template in COMPARISONS.items():
        m = pattern(template).fullmatch(question)
        if m:
            return [(m["s"], (pred,)), (m["o"], (pred,))]
    for pred, templates in QUESTIONS.items():
        for template in templates:
            m = pattern(template).fullmatch(question)
            if m:
                return [chain(m["s"], pred)]
    return None


def chain(subject: str, pred: str) -> Need:
    """ "the city where X lives" asks for a hop through X's city first."""
    for link, template in CHAIN_LINKS.items():
        m = pattern(template).fullmatch(subject)
        if m:
            return (m["s"], (link, pred))
    return (subject, (pred,))


def sentences(body: str) -> list[str]:
    return re.split(r"(?<=\.)\s+", body)


def statements(evidence: Evidence) -> dict[tuple[str, str], str]:
    return {
        (s, p): o
        for hit in evidence
        for sentence in sentences(hit.body)
        for s, p, o in read_sentence(sentence)
    }


def resolved(need: Need, known: dict[tuple[str, str], str]) -> bool:
    subject: str | None = need[0]
    for pred in need[1]:
        subject = known.get((subject, pred))
        if subject is None:
            return False
    return True


class RuleEvaluateModel(KeywordModel):
    name = "rule_evaluate"

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        wanted = needs(framing.question)
        if wanted is None:
            return super().evaluate(framing, evidence)
        known = statements(evidence)
        # Enough by the rule, but no answer to read: nothing may be stated unsupported.
        if all(resolved(n, known) for n in wanted) and self.answer(framing, evidence):
            return Verdict(True)
        return Verdict(False, Exit.NOT_FOUND)
