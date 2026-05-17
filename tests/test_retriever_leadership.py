from retriever import search


def test_retriever_surfaces_opq_for_leadership_queries():
    results = search("We need a leadership assessment for senior executives", top_k=10)
    names = {r.get("name", "").lower() for r in results}
    assert any("opq" in n or "leadership" in n for n in names), f"No OPQ/leadership items in {names}"
