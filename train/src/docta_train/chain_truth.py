"""A chain's bridge, from ground truth: what the core should search next.

"In what year was the city where Orlan Teste lives founded?" bridges through
Varnelle, stated in the source saying Orlan Teste lives there. It is a
training target (where the Read core points, what the Evaluate core judges
after the dig), never a model input.

An answerable chain names its link as its first required fact. An
unanswerable one requires nothing; its link is then the subject's one fact
whose object is of the kind asked ("the city", "the guild"): in these worlds
a subject has one name of each kind. Data building may read the wording; a
model never sees the result as input.
"""

from collections.abc import Iterable

Key = tuple[str, str]  # (world id, question id)
Bridge = tuple[str, str]  # (bridge name, id of a source stating the link)


def chain_bridges(records: Iterable[dict]) -> dict[Key, Bridge]:
    """Every two-hop, trap-free question whose link a source states. Records
    come grouped by world: the world, its sources, then its questions."""
    found: dict[Key, Bridge] = {}
    world, stating = None, {}
    for r in records:
        if r["type"] == "world":
            world, stating = r, {}
        elif r["type"] == "source":
            for fact in r["facts"]:
                stating.setdefault(fact, r["source_id"])
        elif r["hops"] == 2 and not r["trap"]:
            bridge = link_of(r, world, stating)
            if bridge:
                found[(r["world_id"], r["question_id"])] = bridge
    return found


def link_of(question: dict, world: dict, stating: dict[str, str]) -> Bridge | None:
    entities = {e["id"]: e for e in world["entities"]}
    facts = {f["id"]: f for f in world["facts"]}
    if question["answerable"]:
        link = facts[question["required_facts"][0]]
        return entities[link["obj"]]["name"], question["required_sources"][0]
    text = question["text"]
    subjects = [e["id"] for e in entities.values() if e["name"] in text]
    links = [
        f for f in facts.values()
        if f["subj"] in subjects and f["obj"] in entities
        and f"the {entities[f['obj']]['kind']}" in text
    ]  # fmt: skip
    if len(links) != 1 or links[0]["id"] not in stating:
        return None
    return entities[links[0]["obj"]]["name"], stating[links[0]["id"]]
