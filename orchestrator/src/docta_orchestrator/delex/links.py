"""An article's internal links: the names of other pages its text mentions.
They mark which spans are entities without a trained model to find them."""

from docta_orchestrator.realtext.wiki import API


def params(title: str) -> dict:
    # Articles only (namespace 0): categories and files name no entity in prose.
    return {"action": "query", "prop": "links", "titles": title, "plnamespace": "0",
            "pllimit": "max", "redirects": "1", "format": "json", "formatversion": "2"}  # fmt: skip


def of(fetch, title: str) -> list[str]:
    """Every linked title, in the API's order, across all its pages: past 500
    links Wikipedia answers in parts, each naming where the next one starts."""
    found, more = [], {}
    while True:
        answer = fetch.json(API, params(title) | more)
        for page in answer["query"]["pages"]:
            found += [link["title"] for link in page.get("links", [])]
        if "continue" not in answer:
            return found
        more = {"plcontinue": answer["continue"]["plcontinue"]}
