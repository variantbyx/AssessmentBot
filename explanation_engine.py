import re
from typing import Dict, List, Set, Tuple


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text or "")


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", _normalize_text(text))
    return [part.strip() for part in parts if part and part.strip()]


def _clean_fragment(text: str) -> str:
    fragment = _normalize_text(_strip_urls(text))
    fragment = re.sub(r"\s*[-–—]\s*", " ", fragment)
    fragment = re.sub(r"\b(?:this|the|a|an)\s+(?:report|test|assessment|measure|reporting)\b", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\b(?:multi-choice|multiple choice|next-generation|next generation)\b", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\s+", " ", fragment).strip(" .,:;-")
    return fragment


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:] if len(text) > 1 else text.lower()


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z0-9+#.]+", (text or "").lower()))


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in (text or "").lower()


def _has_any(text: str, phrases: List[str]) -> bool:
    lower = (text or "").lower()
    return any(phrase in lower for phrase in phrases)


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


def _capability_phrases_from_description(description: str) -> List[str]:
    text = _normalize_text(_strip_urls(description))
    if not text:
        return []

    text = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", text)
    text = re.sub(r"\([^\)]{0,25}\)", "", text)
    clauses = _sentence_split(text)

    cleaned = []
    prefixes = [
        r"^this report is designed to be given to individuals who have completed the",
        r"^this report is designed to",
        r"^multi-choice test that",
        r"^the next-generation",
        r"^the test",
        r"^this opq report",
        r"^assesses how the candidate",
        r"^provides a detailed analysis of",
        r"^provides",
        r"^measures the knowledge of",
        r"^measures",
        r"^tests",
        r"^evaluates",
    ]

    for clause in clauses:
        candidate = _clean_fragment(clause)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        lower_candidate = candidate.lower()

        for pattern in prefixes:
            candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
            lower_candidate = candidate.lower()

        candidate = re.sub(r"\s*[:;,-]\s*", " ", candidate).strip(" .,:;-")
        candidate = re.sub(r"\b(?:assess|benchmark|evaluate|measure)\s+(?:your|the|candidate's|candidate)\b", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\b(?:assess|benchmark|evaluate|measure)\b", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;-")

        if len(candidate) >= 24:
            cleaned.append(candidate)

    if cleaned:
        return cleaned

    fallback = _clean_fragment(text)
    return [fallback] if fallback else []


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
    clauses = _capability_phrases_from_description(description)
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
        return _clean_fragment(scored[0][1])

    return _clean_fragment(clauses[0])


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


def _assessment_style(item: Dict[str, object], query_profile: Dict[str, object]) -> str:
    text = _get_item_text(item)
    categories = [str(category).lower() for category in (item.get("categories", []) or [])]

    if query_profile.get("seniority") == "graduate" and _has_any(text + " " + " ".join(categories), ["ability", "aptitude", "reasoning", "numerical", "cognitive"]):
        return "graduate"
    if "aptitude" in query_profile.get("matched_intents", []) and _has_any(text + " " + " ".join(categories), ["ability", "aptitude", "reasoning", "numerical", "cognitive"]):
        return "aptitude"
    if _has_any(text, ["simulation", "live coding", "automata", "coding"]):
        return "coding"
    if _has_any(text, ["leadership", "people management", "manager", "team lead", "opq"]):
        return "leadership"
    if _has_any(text, ["personality", "behavior", "behaviour", "traits"]):
        return "behavioral"
    if _has_any(text, ["reasoning", "numerical", "cognitive", "logical", "aptitude"]):
        return "aptitude"
    if _has_any(text, ["communication", "verbal", "interpersonal", "presentation"]):
        return "communication"
    if _has_any(text, ["java", "spring", "python", "aws", "sql", "backend", "framework"]):
        return "technical"
    if query_profile.get("seniority") == "graduate":
        return "graduate"
    return "general"


def _intent_bridge_phrase(query_profile: Dict[str, object], style: str, capability_hits: List[str]) -> str:
    intents = query_profile.get("matched_intents", []) or []
    role_phrases = query_profile.get("role_phrases", []) or []
    seniority = query_profile.get("seniority")
    
    if style in ["aptitude", "graduate"]:
        if seniority == "graduate" or style == "graduate":
            return "That makes it a strong fit for early-career screening and analytical potential checks."
        return "That makes it a strong fit for reasoning-focused screening and candidate potential checks."

    if intents:
        intent_text = _join_list(intents[:3])
        templates = [
            f"It aligns closely with your focus on {intent_text}.",
            f"That makes it a strong fit for {intent_text} screening.",
            f"It maps directly to the hiring intent behind {intent_text}.",
        ]
        seed = len(capability_hits) + len(role_phrases) + (1 if seniority else 0)
        return templates[seed % len(templates)]

    if role_phrases:
        role_text = _join_list(role_phrases[:2])
        return f"That makes it a practical option for {role_text} hiring workflows."

    if style in ["aptitude", "graduate"]:
        return "That makes it a practical fit for early-funnel screening and candidate potential checks."

    return "That makes it a practical fit for shortlist decisions and interview planning."


def _opening_phrase(name: str, style: str, seed: int) -> str:
    pools = {
        "technical": [
            f"{name} is particularly effective for technical hiring",
            f"{name} aligns well with backend and engineering screening",
            f"{name} is highly suitable for framework-level evaluation",
            f"{name} works especially well for technical assessment workflows",
        ],
        "coding": [
            f"{name} works especially well for hands-on coding evaluation",
            f"{name} is particularly effective when hiring for practical programming ability",
            f"{name} is highly suitable for software engineering screening",
            f"{name} is designed to assess real-world coding performance",
        ],
        "leadership": [
            f"{name} is particularly effective for leadership hiring",
            f"{name} aligns well with manager-level evaluation",
            f"{name} is highly suitable for leadership readiness screening",
            f"{name} works especially well for people-management assessment",
        ],
        "behavioral": [
            f"{name} is particularly effective for behavioral assessment",
            f"{name} is highly suitable for workplace personality screening",
            f"{name} aligns well with organizational fit evaluation",
            f"{name} works especially well for behavioral hiring workflows",
        ],
        "aptitude": [
            f"{name} is particularly effective for aptitude screening",
            f"{name} aligns well with early-funnel cognitive assessment",
            f"{name} is highly suitable for analytical potential evaluation",
            f"{name} works especially well for graduate candidate screening",
        ],
        "communication": [
            f"{name} is particularly effective for communication-focused hiring",
            f"{name} aligns well with interpersonal skill evaluation",
            f"{name} is highly suitable for stakeholder communication screening",
            f"{name} works especially well for verbal capability assessment",
        ],
        "graduate": [
            f"{name} is particularly effective for early-career hiring",
            f"{name} aligns well with graduate candidate screening",
            f"{name} is highly suitable for entry-level evaluation",
            f"{name} works especially well for identifying high-potential graduates",
        ],
        "general": [
            f"{name} is particularly effective for this hiring need",
            f"{name} aligns well with the role requirements",
            f"{name} is highly suitable for this assessment use case",
            f"{name} works especially well for shortlist screening",
        ],
    }

    options = pools.get(style, pools["general"])
    return options[seed % len(options)]


def _transition_phrase(style: str, seed: int) -> str:
    pools = {
        "technical": [
            "because it evaluates",
            "since it measures",
            "as it covers",
            "because it is designed to assess",
        ],
        "coding": [
            "because it measures",
            "since it evaluates",
            "as it provides",
            "because it is designed to assess",
        ],
        "leadership": [
            "because it evaluates",
            "since it measures",
            "as it captures",
            "because it is designed to assess",
        ],
        "behavioral": [
            "because it evaluates",
            "since it captures",
            "as it measures",
            "because it is designed to assess",
        ],
        "aptitude": [
            "because it measures",
            "since it evaluates",
            "as it captures",
            "because it is designed to assess",
        ],
        "communication": [
            "because it evaluates",
            "since it measures",
            "as it captures",
            "because it is designed to assess",
        ],
        "graduate": [
            "because it measures",
            "since it evaluates",
            "as it captures",
            "because it is designed to assess",
        ],
        "general": [
            "because it evaluates",
            "since it measures",
            "as it captures",
            "because it is designed to assess",
        ],
    }
    options = pools.get(style, pools["general"])
    return options[seed % len(options)]


def _capability_summary(style: str, capability_hits: List[str], evidence: str, categories: List[str]) -> str:
    evidence_text = _clean_fragment(evidence)
    evidence_text = re.sub(r"\b(?:assess|benchmark|evaluate|measure)\b", "", evidence_text, flags=re.IGNORECASE)
    evidence_text = re.sub(r"\s+", " ", evidence_text).strip(" .,:;-")
    
    if style in ["aptitude", "graduate"]:
        aptitude_hits = [hit for hit in capability_hits if hit in ["aptitude", "reasoning", "numerical", "cognitive", "logical", "deductive", "inductive"]]
        if aptitude_hits:
            return f"It measures { _join_list(aptitude_hits[:3]) } and analytical potential."
        return "It measures reasoning ability, analytical potential, and trainability."

    if style == "coding":
        if capability_hits:
            return f"It measures practical programming ability across { _join_list(capability_hits[:3]) }."
        return "It measures real-world programming performance through hands-on coding tasks."

    if style == "technical":
        if capability_hits:
            return f"It evaluates { _join_list(capability_hits[:3]) } knowledge relevant to engineering roles."
        return "It evaluates technical knowledge that supports framework-level screening."

    if style == "leadership":
        if capability_hits:
            return f"It evaluates { _join_list(capability_hits[:3]) } and leadership readiness."
        return "It evaluates leadership potential, decision-making tendencies, and people-management readiness."

    if style == "behavioral":
        aptitude_hits = [hit for hit in capability_hits if hit in ["aptitude", "reasoning", "numerical", "cognitive", "logical", "deductive", "inductive"]]
        if aptitude_hits:
            return f"It measures { _join_list(aptitude_hits[:3]) } ability and analytical potential."
        return "It measures cognitive potential, reasoning ability, and trainability."

    if style == "aptitude":
        if capability_hits:
            return f"It measures { _join_list(capability_hits[:3]) } ability and analytical potential."
        return "It measures cognitive potential, reasoning ability, and trainability."

    if style == "communication":
        if capability_hits:
            return f"It assesses { _join_list(capability_hits[:3]) } skill and stakeholder communication."
        return "It assesses verbal clarity, interpersonal skill, and business communication effectiveness."

    if style == "graduate":
        if capability_hits:
            return f"It measures { _join_list(capability_hits[:3]) } capability for early-career screening."
        return "It measures early-career potential, reasoning ability, and hiring readiness."

    if evidence_text:
        return f"It evaluates {evidence_text.lower()}."
    if categories:
        return f"Its focus on { _join_list(categories[:3]) } supports role-aligned screening."
    return "It supports role-aligned screening and shortlist decisions."


def _hiring_suitability_phrase(style: str, query_profile: Dict[str, object], capability_hits: List[str], categories: List[str], job_levels: List[str], assessment_type: str) -> str:
    seniority = query_profile.get("seniority")
    has_graduate_level = any("graduate" in str(level).lower() or "entry" in str(level).lower() for level in job_levels)
    lower_categories = [str(category).lower() for category in categories]

    if style == "coding":
        return "Strong fit for hands-on coding assessment and practical programming evaluation."
    if style == "technical":
        if capability_hits:
            return f"Useful for backend engineering screening and { _join_list(capability_hits[:2]) } evaluation."
        return "Useful for backend engineering screening and framework-level technical evaluation."
    if style == "leadership":
        return "Useful for leadership readiness assessment and people-management evaluation."
    if style == "behavioral":
        return "Suitable for assessing workplace personality and organizational fit."
    if style == "aptitude":
        if seniority == "graduate" or has_graduate_level:
            return "Well suited for early-career candidate screening and cognitive potential evaluation."
        return "Useful for analytical potential screening and reasoning-based hiring decisions."
    if style == "communication":
        return "Useful for communication skill screening and stakeholder effectiveness evaluation."
    if style == "graduate":
        return "Well suited for early-career candidate screening and cognitive potential evaluation."

    if assessment_type == "Coding Simulation":
        return "Strong fit for hands-on coding assessment and practical programming evaluation."
    if assessment_type == "Aptitude / Reasoning":
        return "Well suited for early-career candidate screening and cognitive potential evaluation."
    if assessment_type == "Personality / Behavioral":
        return "Suitable for assessing workplace personality and organizational fit."
    if _has_any(" ".join(lower_categories), ["knowledge", "skills", "framework", "simulation"]):
        return "Useful for technical screening and role-specific capability evaluation."
    return "Useful for role-aligned screening and shortlist decisions."


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
        job_levels = item.get("job_levels", []) or []
        capability_hits = _capability_matches(item, query_profile)
        evidence = _best_description_evidence(item, query_profile, capability_hits)
        style = _assessment_style(item, query_profile)
        seed = sum(ord(c) for c in str(name)) + idx + len(query_profile.get("matched_intents", []))

        opener = _opening_phrase(name, style, seed)
        capability_summary = _capability_summary(style, capability_hits, evidence, categories)
        suitability = _hiring_suitability_phrase(style, query_profile, capability_hits, categories, job_levels, item.get("assessment_type", ""))
        bridge = _intent_bridge_phrase(query_profile, style, capability_hits)

        if categories:
            category_hint = f"Its primary focus sits within {_join_list(categories[:2])}."
        else:
            category_hint = "Its scope is aligned to the hiring intent in your query."

        explanation = " ".join([
            opener + ".",
            bridge,
            capability_summary,
            category_hint,
            suitability,
        ])

        explanation = re.sub(r"\s+", " ", explanation).strip()
        explanation = explanation.replace(" .", ".")
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
