import re
from typing import Dict, List, Set, Tuple


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z0-9+#.]+", (text or "").lower()))


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in (text or "").lower()


QUERY_SIGNALS: Dict[str, List[str]] = {
    "java": ["java", "spring", "hibernate", "jdbc"],
    "python": ["python", "django", "flask", "pandas"],
    "aws": ["aws", "cloud", "ec2", "lambda"],
    "sql": ["sql", "database", "data modeling"],
    "coding": ["coding", "programming", "developer", "software engineer", "live coding"],
    "leadership": ["leadership", "people management", "manager", "team lead"],
    "communication": ["communication", "verbal", "interpersonal", "business communication"],
    "personality": ["personality", "behavioral", "behavioural", "traits"],
    "aptitude": ["aptitude", "cognitive", "reasoning", "numerical", "logical"],
    "graduate": ["graduate", "entry-level", "entry level", "fresher", "campus"],
    "sales": ["sales", "account", "customer"],
    "backend": ["backend", "api", "services", "microservices"],
}


CAPABILITY_HINTS: Dict[str, List[str]] = {
    "java": ["java", "spring", "hibernate", "jdbc", "enterprise java"],
    "python": ["python", "programming", "coding"],
    "coding": ["coding", "programming", "simulation", "hands-on", "live coding"],
    "leadership": ["leadership", "people management", "decision", "managerial"],
    "communication": ["communication", "verbal", "interpersonal", "presentation"],
    "aptitude": ["deductive", "inductive", "numerical", "reasoning", "analytical"],
    "personality": ["personality", "traits", "behavior", "behaviour"],
    "sales": ["sales", "negotiation", "customer", "influence"],
    "backend": ["backend", "api", "services", "architecture"],
    "aws": ["cloud", "aws", "deployment", "security"],
    "sql": ["sql", "database", "data", "modeling"],
}


INTENT_TO_HIRING_USE = {
    "coding": "validate practical coding performance before technical interviews",
    "java": "screen Java-specific readiness for backend engineering roles",
    "python": "evaluate Python programming ability for software development workflows",
    "aptitude": "identify analytical potential and trainability early in the funnel",
    "leadership": "assess people-management potential and leadership readiness",
    "communication": "assess clarity, interpersonal effectiveness, and business communication",
    "personality": "understand workplace style and behavioral fit for the team",
    "graduate": "differentiate high-potential early-career candidates",
    "sales": "evaluate customer-facing effectiveness for commercial roles",
}


FOCUS_FROM_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("coding simulation", ["simulation", "live coding", "hands-on", "programming tasks", "automata"]),
    ("technical knowledge", ["knowledge", "framework", "architecture", "concepts", "fundamentals"]),
    ("aptitude reasoning", ["deductive", "inductive", "numerical", "reasoning", "cognitive"]),
    ("leadership potential", ["leadership", "management", "people", "decision"]),
    ("behavioral fit", ["personality", "behavior", "behaviour", "traits"]),
    ("communication effectiveness", ["communication", "verbal", "interpersonal", "presentation"]),
]


def extract_query_profile(query: str) -> Dict[str, object]:
    q = _normalize_text(query)
    q_low = q.lower()
    tokens = _tokenize(q)

    matched_intents: List[str] = []
    matched_terms: List[str] = []

    for intent, phrases in QUERY_SIGNALS.items():
        for phrase in phrases:
            if phrase in q_low:
                matched_intents.append(intent)
                matched_terms.append(phrase)
                break

    role_phrases = []
    for role_hint in [
        "developer",
        "engineer",
        "manager",
        "analyst",
        "sales",
        "backend",
        "frontend",
        "leader",
    ]:
        if role_hint in q_low:
            role_phrases.append(role_hint)

    seniority = None
    if "graduate" in q_low or "entry level" in q_low or "entry-level" in q_low:
        seniority = "graduate"
    elif "senior" in q_low:
        seniority = "senior"
    elif "manager" in q_low:
        seniority = "manager"

    return {
        "query": q,
        "tokens": tokens,
        "matched_intents": matched_intents,
        "matched_terms": matched_terms,
        "role_phrases": role_phrases,
        "seniority": seniority,
        "is_comparison": "compare" in q_low,
    }


def _get_item_text(item: Dict[str, object]) -> str:
    return " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description", "")),
            " ".join(item.get("categories", []) or []),
            " ".join(item.get("job_levels", []) or []),
        ]
    ).lower()


def _extract_description_clauses(description: str) -> List[str]:
    clean = _normalize_text(description)
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [p.strip() for p in parts if len(p.strip()) >= 35]


def _capability_matches(item: Dict[str, object], query_profile: Dict[str, object]) -> List[str]:
    item_text = _get_item_text(item)
    matched: List[str] = []

    for intent in query_profile["matched_intents"]:
        hints = CAPABILITY_HINTS.get(intent, [])
        for hint in hints:
            if hint in item_text and hint not in matched:
                matched.append(hint)

    for generic in ["reasoning", "numerical", "deductive", "inductive", "coding", "simulation", "leadership", "communication"]:
        if generic in item_text and generic not in matched:
            matched.append(generic)

    return matched[:6]


def _best_description_evidence(item: Dict[str, object], query_profile: Dict[str, object], capability_hits: List[str]) -> str:
    description = item.get("description", "") or ""
    clauses = _extract_description_clauses(description)
    if not clauses:
        return ""

    intent_terms = list(query_profile["matched_terms"]) + capability_hits
    scored = []

    for clause in clauses:
        low = clause.lower()
        score = sum(1 for t in intent_terms if t and t in low)
        scored.append((score, clause))

    scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

    if scored and scored[0][0] > 0:
        return scored[0][1]

    return clauses[0]


def _role_fit_phrase(query_profile: Dict[str, object], item: Dict[str, object]) -> str:
    job_levels = item.get("job_levels", []) or []
    role_phrases = query_profile.get("role_phrases", []) or []
    seniority = query_profile.get("seniority")

    if seniority == "graduate":
        if any("graduate" in lvl.lower() or "entry" in lvl.lower() for lvl in job_levels):
            return "It aligns well with graduate and entry-level hiring"
        return "It supports early-career talent screening"

    if seniority == "senior":
        if any(any(tag in lvl.lower() for tag in ["mid", "manager", "professional", "director"]) for lvl in job_levels):
            return "It fits experienced-candidate evaluation"

    if "manager" in role_phrases or seniority == "manager":
        return "It is suitable for manager-track selection decisions"

    if role_phrases:
        return f"It is relevant for {' '.join(role_phrases[:2])} hiring"

    return "It is relevant to the role requirements in your query"


def _join_list(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _vary(index_seed: int, options: List[str]) -> str:
    if not options:
        return ""
    return options[index_seed % len(options)]


def generate_recommendation_explanations(query: str, recommendations: List[Dict[str, object]], limit: int = 3) -> str:
    query_profile = extract_query_profile(query)
    outputs: List[str] = []

    for idx, item in enumerate((recommendations or [])[:limit]):
        name = item.get("name", "This assessment")
        categories = item.get("categories", []) or []
        capability_hits = _capability_matches(item, query_profile)
        evidence = _best_description_evidence(item, query_profile, capability_hits)
        role_fit = _role_fit_phrase(query_profile, item)

        seed = sum(ord(c) for c in str(name)) + idx
        opener = _vary(seed, [
            f"{name} stands out for this hiring need",
            f"{name} is a strong match for your requirement",
            f"{name} is recommended for this role profile",
        ])

        if capability_hits:
            skill_reason = (
                "It matches your focus on "
                f"{_join_list(capability_hits[:4])}."
            )
        elif query_profile["matched_intents"]:
            intents = _join_list(query_profile["matched_intents"][:3])
            skill_reason = f"It aligns with your stated focus on {intents}."
        elif categories:
            skill_reason = (
                "Its assessment focus is "
                f"{_join_list(categories[:3])}, which supports this search."
            )
        else:
            skill_reason = "It aligns with the role and capability needs in your query."

        if evidence:
            evidence_sentence = f"From the assessment scope: {evidence}"
        else:
            evidence_sentence = "Its structure supports targeted screening decisions for this role."

        hiring_uses = [INTENT_TO_HIRING_USE[i] for i in query_profile["matched_intents"] if i in INTENT_TO_HIRING_USE]
        if hiring_uses:
            hiring_sentence = f"In practice, it helps recruiters {_join_list(hiring_uses[:2])}."
        else:
            hiring_sentence = "In practice, it supports clearer shortlist decisions and interview planning."

        explanation = " ".join([opener + ".", role_fit + ".", skill_reason, evidence_sentence, hiring_sentence])
        outputs.append(explanation)

    if outputs:
        return "\n\n".join(outputs)

    return "I could not find suitable assessments for the current query. Please share role, skills, and hiring intent for better recommendations."


def _infer_focus(item: Dict[str, object]) -> str:
    text = _get_item_text(item)
    scores: List[Tuple[int, str]] = []

    for label, cues in FOCUS_FROM_KEYWORDS:
        score = sum(1 for cue in cues if cue in text)
        scores.append((score, label))

    scores.sort(reverse=True)
    if scores and scores[0][0] > 0:
        return scores[0][1]

    categories = item.get("categories", []) or []
    if categories:
        return categories[0].lower()

    return "general role fit"


def _hiring_usage_from_focus(focus: str) -> str:
    mapping = {
        "coding simulation": "validating real-world coding execution",
        "technical knowledge": "screening knowledge depth before technical rounds",
        "aptitude reasoning": "shortlisting candidates with strong analytical potential",
        "leadership potential": "identifying leadership readiness and management potential",
        "behavioral fit": "evaluating cultural and behavioral alignment",
        "communication effectiveness": "assessing stakeholder communication capability",
    }
    return mapping.get(focus, "supporting role-specific screening decisions")


def generate_comparison_explanation(query: str, first: Dict[str, object], second: Dict[str, object]) -> str:
    query_profile = extract_query_profile(query)

    first_name = first.get("name", "First assessment")
    second_name = second.get("name", "Second assessment")

    first_focus = _infer_focus(first)
    second_focus = _infer_focus(second)

    first_caps = _capability_matches(first, query_profile)
    second_caps = _capability_matches(second, query_profile)

    first_evidence = _best_description_evidence(first, query_profile, first_caps)
    second_evidence = _best_description_evidence(second, query_profile, second_caps)

    first_use = _hiring_usage_from_focus(first_focus)
    second_use = _hiring_usage_from_focus(second_focus)

    first_strength = _join_list(first_caps[:3]) if first_caps else first_focus
    second_strength = _join_list(second_caps[:3]) if second_caps else second_focus

    lines = [
        f"{first_name} and {second_name} serve different evaluation goals.",
        f"{first_name} emphasizes {first_focus} with stronger coverage of {first_strength}, whereas {second_name} emphasizes {second_focus} with stronger coverage of {second_strength}.",
        f"For hiring decisions, {first_name} is better suited to {first_use}, while {second_name} is better suited to {second_use}.",
    ]

    if first_evidence:
        lines.append(f"{first_name} evidence: {first_evidence}")
    if second_evidence:
        lines.append(f"{second_name} evidence: {second_evidence}")

    if query_profile["matched_intents"]:
        lines.append(
            "Given your query focus on "
            f"{_join_list(query_profile['matched_intents'][:3])}, "
            "choose based on whether you want knowledge depth, behavior signals, or practical execution evidence."
        )

    return " ".join(lines)
