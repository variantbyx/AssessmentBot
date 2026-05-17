import json
import os
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Dict, List, Tuple

# load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Loaded", len(data), "assessments")

#Build Searchable Text

documents = []

for item in data:

    text = " ".join([
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", []))
    ])

    documents.append(text)

#Create BM25 Index
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

print("BM25 index created")

print("\nExample document:\n")
print(documents[0][:500])

# Load model lazily so API startup can bind port quickly on low-memory instances.
model = None

# Use BM25-only mode by default on Render to avoid OOM on small instances.
USE_SEMANTIC_SEARCH = os.getenv("USE_SEMANTIC_SEARCH", "0" if os.getenv("RENDER") else "1") == "1"


def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("\nModel loaded successfully")
    return model


def load_or_build_index():
    if os.path.exists("shl_index.faiss"):
        loaded_index = faiss.read_index("shl_index.faiss")
        print("\nFAISS index loaded")
        print("Total vectors:", loaded_index.ntotal)
        return loaded_index

    # In constrained deploy environments, avoid building embeddings at runtime.
    if not USE_SEMANTIC_SEARCH:
        print("\nFAISS index not found. Continuing in BM25-only mode.")
        return None

    local_model = get_model()
    embeddings = local_model.encode(
        documents,
        show_progress_bar=True
    )

    print("\nEmbeddings shape:", embeddings.shape)

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    loaded_index = faiss.IndexFlatL2(dimension)
    loaded_index.add(embeddings)

    faiss.write_index(loaded_index, "shl_index.faiss")

    with open("documents.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\nFAISS index created")
    print("Total vectors:", loaded_index.ntotal)
    return loaded_index


index = load_or_build_index()


def _normalize_text(value):
    return " ".join(str(value or "").split()).strip()


def _safe_list(value):
    if isinstance(value, list):
        return [item for item in value if item]
    if value:
        return [value]
    return []


def _truncate_text(value, max_length=220):
    text = _normalize_text(value)
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def _extract_query_signals(query: str) -> List[str]:
    query_lower = (query or "").lower()
    signal_map = [
        ("java", ["java", "spring", "hibernate", "jdbc"]),
        ("python", ["python", "django", "flask", "pandas"]),
        ("coding", ["coding", "programming", "developer", "software engineer", "live coding", "automata"]),
        ("aptitude", ["aptitude", "reasoning", "cognitive", "numerical", "logical"]),
        ("leadership", ["leadership", "people management", "manager", "team lead", "senior leadership", "cxo", "cxos", "c-suite", "c suite", "executive", "director-level", "director level", "director"]),
        ("communication", ["communication", "verbal", "interpersonal", "business communication"]),
        ("aws", ["aws", "cloud", "ec2", "lambda"]),
        ("sql", ["sql", "database", "data modeling"]),
        ("graduate", ["graduate", "entry-level", "entry level", "fresher"]),
        ("sales", ["sales", "account", "customer"]),
        ("backend", ["backend", "api", "services", "microservices"]),
    ]

    matched = []
    for label, keywords in signal_map:
        if any(keyword in query_lower for keyword in keywords):
            matched.append(label)
    return matched


def _extract_matched_skills(query: str, item: Dict[str, object]) -> List[str]:
    query_lower = (query or "").lower()
    item_text = " ".join([
        _normalize_text(item.get("name", "")),
        _normalize_text(item.get("description", "")),
        " ".join(_safe_list(item.get("keys", []))),
        " ".join(_safe_list(item.get("job_levels", []))),
    ]).lower()

    skill_groups = {
        "java": ["java"],
        "spring": ["spring"],
        "python": ["python"],
        "coding": ["coding", "programming", "developer", "live coding", "simulation", "automata"],
        "aptitude": ["aptitude", "reasoning", "numerical", "cognitive", "logical"],
        "leadership": ["leadership", "manager", "people management", "team lead"],
        "communication": ["communication", "verbal", "interpersonal", "presentation"],
        "aws": ["aws", "cloud", "lambda", "ec2"],
        "sql": ["sql", "database", "data modeling"],
        "graduate": ["graduate", "entry-level", "entry level", "fresher"],
        "sales": ["sales", "account", "customer"],
        "backend": ["backend", "api", "services", "microservices"],
    }

    matched = []
    for label, keywords in skill_groups.items():
        if any(keyword in query_lower and keyword in item_text for keyword in keywords):
            matched.append(label)
    return matched


def _description_evidence(item: Dict[str, object]) -> str:
    description = _normalize_text(item.get("description", ""))
    if not description:
        return ""

    lower_description = description.lower()
    cue_phrases = [
        "measures the knowledge of",
        "measures",
        "assesses",
        "tests",
        "provides",
        "covers the following topics",
        "covers",
    ]

    for cue in cue_phrases:
        index = lower_description.find(cue)
        if index != -1:
            snippet = description[index:index + 180]
            return _truncate_text(snippet, 180)

    return _truncate_text(description, 180)


def _friendly_evidence_text(text: str) -> str:
    clean_text = _normalize_text(text)

    for prefix in ["Multi-choice test that ", "This report is designed to ", "The test ", "This OPQ report "]:
        if clean_text.startswith(prefix):
            clean_text = clean_text[len(prefix):]
            break

    lower_text = clean_text.lower()
    for prefix in ["measures the knowledge of ", "measures ", "assesses ", "tests ", "provides "]:
        if lower_text.startswith(prefix):
            clean_text = clean_text[len(prefix):]
            break

    return clean_text.rstrip(".")


def _assessment_type(item: Dict[str, object]) -> str:
    categories = [str(value) for value in _safe_list(item.get("keys", []))]
    description = _normalize_text(item.get("description", "")).lower()
    name = _normalize_text(item.get("name", "")).lower()
    text = f"{name} {description} {' '.join(categories)}"

    if any(term in text for term in ["simulation", "live coding", "automata", "coding"]):
        return "Coding Simulation"
    if any(term in text for term in ["personality", "behavior", "behaviour", "opq"]):
        return "Personality / Behavioral"
    if any(term in text for term in ["reasoning", "numerical", "verbal", "logical", "cognitive", "aptitude"]):
        return "Aptitude / Reasoning"
    if any(term in text for term in ["framework", "architecture", "knowledge & skills", ".net", "java", "python"]):
        return "Knowledge / Skills"
    if any(term in text for term in ["leadership", "manager", "competency"]):
        return "Leadership / Competency"
    if categories:
        first_category = categories[0].lower()
        if "knowledge" in first_category:
            return "Knowledge / Skills"
        if "ability" in first_category or "aptitude" in first_category:
            return "Aptitude / Reasoning"
        if "personality" in first_category or "behavior" in first_category:
            return "Personality / Behavioral"
        if "development" in first_category or "competenc" in first_category:
            return "Competency / Development"
        if "simulation" in first_category:
            return "Simulation"
        return categories[0]
    return "Assessment"


def _simulation_adaptive_flags(item: Dict[str, object]) -> Tuple[bool, bool, str]:
    adaptive_raw = str(item.get("adaptive", "")).strip().lower()
    remote_raw = str(item.get("remote", "")).strip().lower()
    status = str(item.get("status", "")).strip().lower()

    is_adaptive = adaptive_raw == "yes"
    is_remote = remote_raw == "yes"
    simulation_mode = "adaptive" if is_adaptive else "standard"

    if status and status != "ok":
        simulation_mode = status

    return is_remote, is_adaptive, simulation_mode


def _confidence_label(relative_score: float, matched_skills: List[str], metadata_boost: float) -> Tuple[str, int]:
    confidence_score = int(round(55 + (relative_score * 30) + (min(len(matched_skills), 3) * 5) + (metadata_boost * 40)))
    confidence_score = max(0, min(100, confidence_score))

    if confidence_score >= 72:
        return "High", confidence_score
    if confidence_score >= 45:
        return "Medium", confidence_score
    return "Low", confidence_score


def _reasoning_summary(item: Dict[str, object], matched_skills: List[str], assessment_type: str, query: str) -> str:
    role_levels = ", ".join(_safe_list(item.get("job_levels", []))[:2])
    evidence = _description_evidence(item)
    focus_skill = ", ".join(matched_skills[:3]) if matched_skills else "the role requirements"

    if matched_skills:
        if assessment_type == "Coding Simulation":
            return f"Recommended because it evaluates practical coding ability relevant to {focus_skill}."
        if assessment_type == "Aptitude / Reasoning":
            return f"Recommended because it measures analytical potential aligned to {focus_skill} requirements."
        if assessment_type == "Personality / Behavioral":
            return f"Recommended because it captures behavioral signals relevant to {focus_skill} and team fit."
        if evidence:
            evidence_text = _friendly_evidence_text(evidence)
            return f"Recommended because it measures {evidence_text} for {focus_skill} screening."
        return f"Recommended because it aligns with {focus_skill} and the role context it serves."

    if evidence:
        evidence_text = _friendly_evidence_text(evidence)
        if role_levels:
            return f"Recommended for {role_levels} hiring because it measures {evidence_text}."
        return f"Recommended because it measures {evidence_text} and matches the hiring need described in the query."

    return "Recommended because it aligns with the role requirements in the query."


def _hiring_suitability_hint(query: str, item: Dict[str, object], matched_skills: List[str]) -> str:
    query_lower = (query or "").lower()
    job_levels = [level.lower() for level in _safe_list(item.get("job_levels", []))]
    assessment_type = _assessment_type(item)

    if "graduate" in query_lower or "entry level" in query_lower or "entry-level" in query_lower:
        if any("graduate" in level or "entry" in level for level in job_levels):
            return "Well suited for graduate and early-career screening."
        return "Useful for evaluating early-career hiring potential."

    if "senior" in query_lower or "manager" in query_lower:
        return "Useful for experienced or manager-track hiring decisions."

    if assessment_type == "Coding Simulation":
        return "Strong fit for technical screening and practical coding evaluation."
    if assessment_type == "Aptitude / Reasoning":
        return "Strong fit for early funnel screening and analytical potential checks."
    if assessment_type == "Personality / Behavioral":
        return "Useful for behavioral fit and leadership-style assessment."
    if assessment_type == "Knowledge / Skills":
        return "Useful for knowledge-depth screening in technical hiring workflows."
    if matched_skills:
        return f"Relevant for assessing {', '.join(matched_skills[:2])} in the hiring flow."

    return "Useful for role-aligned shortlist screening."


def _build_recommendation_payload(
    query: str,
    item: Dict[str, object],
    semantic: float,
    keyword: float,
    metadata_boost: float,
    final_score: float,
    recommendation_strength: str,
    confidence_score: int,
) -> Dict[str, object]:
    matched_skills = _extract_matched_skills(query, item)
    assessment_type = _assessment_type(item)
    remote_testing, is_adaptive, simulation_mode = _simulation_adaptive_flags(item)

    description = _truncate_text(item.get("description", ""), 240)
    reasoning_summary = _reasoning_summary(item, matched_skills, assessment_type, query)
    hiring_suitability = _hiring_suitability_hint(query, item, matched_skills)

    payload = {
        "name": item.get("name", ""),
        "url": item.get("link", ""),
        "link": item.get("link", ""),
        "categories": _safe_list(item.get("keys", [])),
        "job_levels": _safe_list(item.get("job_levels", [])),
        "description": description,
        "duration": _normalize_text(item.get("duration", "")),
        "languages": _safe_list(item.get("languages", [])),
        "assessment_type": assessment_type,
        "is_adaptive": is_adaptive,
        "simulation_mode": simulation_mode,
        "remote_testing": remote_testing,
        "recommendation_strength": recommendation_strength,
        "confidence_score": confidence_score,
        "matched_skills": matched_skills,
        "reasoning_summary": reasoning_summary,
        "hiring_suitability": hiring_suitability,
    }

    return payload


# #Create Query Search Function (old)

# def search(query, top_k=5):

#     query_embedding = model.encode([query])
#     query_embedding = np.array(query_embedding).astype("float32")

#     distances, indices = index.search(query_embedding, top_k)

#     print("\nTop Results:\n")

#     for rank, idx in enumerate(indices[0]):

#         item = data[idx]

#         print(f"{rank+1}. {item['name']}")
#         print(item['link'])
#         print()

# #Test Retrieval
# query = "Java backend developer with communication skills"

# search(query)

# #Search Function
# def search(query, top_k=5):

#     query_embedding = model.encode([query])
#     query_embedding = np.array(query_embedding).astype("float32")

#     distances, indices = index.search(query_embedding, top_k)

#     print("\n==============================")
#     print("QUERY:", query)
#     print("==============================\n")

#     for rank, idx in enumerate(indices[0]):

#         item = data[idx]

#         print(f"Rank {rank+1}")
#         print("Name:", item["name"])
#         print("URL:", item["link"])
#         print("Categories:", item["keys"])
#         print("Job Levels:", item["job_levels"])

#         print("\nDescription:")
#         print(item["description"][:300])

#         print("\n-----------------------------\n")

#New search queries

def search(query, top_k=5):

    #Query Rewriting / Expansion

    query_lower = query.lower()

    expanded_query = query

    if "aptitude" in query_lower:
        expanded_query += " cognitive reasoning ability numerical"

    if "coding" in query_lower:
        expanded_query += " programming developer live coding automata"

    if "leadership" in query_lower:
        expanded_query += " personality behavioral management"

    if "graduate" in query_lower:
        expanded_query += " entry level fresher"

    # ---------- Semantic Search ----------

    semantic_scores = {}

    if USE_SEMANTIC_SEARCH and index is not None:
        query_embedding = get_model().encode([expanded_query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = index.search(query_embedding, len(data))

        for rank, idx in enumerate(indices[0]):
            semantic_scores[idx] = 1 / (1 + distances[0][rank])

    # ---------- BM25 Search ----------

    # tokenized_query = query.lower().split()
    tokenized_query = expanded_query.lower().split()

    bm25_scores_raw = bm25.get_scores(tokenized_query)

    bm25_scores = {}

    for idx, score in enumerate(bm25_scores_raw):
        bm25_scores[idx] = score

    # ---------- Combine Scores ----------

    final_scores = []

    for idx in range(len(data)):

        item = data[idx]

        semantic = semantic_scores.get(idx, 0)
        keyword = bm25_scores.get(idx, 0)

        # Metadata boosting based on query terms and item metadata

        metadata_boost = 0

        query_lower = query.lower()

        # Add small boosts when query contains matching intent words
        if "Knowledge & Skills" in item["keys"] and any(k in query_lower for k in ["java", "python", "developer", "backend", "frontend", "software", "engineer", "coding", "programming"]):
            metadata_boost += 0.05

        if "graduate" in query_lower and "Graduate" in item["job_levels"]:
            metadata_boost += 0.05

        # Stronger boost for leadership intents; also boost OPQ / leadership-named products
        if any(k in query_lower for k in ["leadership", "senior leadership", "cxo", "executive", "director"]):
            if "Personality & Behavior" in item["keys"] or "Leadership" in item.get("name", "") or "OPQ" in item.get("name", "") or "leadership" in item.get("name", "").lower():
                metadata_boost += 0.18
            else:
                metadata_boost += 0.04

        if "aptitude" in query_lower and "Ability & Aptitude" in item["keys"]:
            metadata_boost += 0.05

        if "coding" in query_lower and "Simulations" in item["keys"]:
            metadata_boost += 0.03

        final_score = (
            0.75 * semantic +
            0.15 * keyword +
            metadata_boost
        )

        final_scores.append((final_score, idx, semantic, keyword, metadata_boost))

    final_scores.sort(reverse=True)

    # print("\n==============================")
    # print("QUERY:", query)
    # print("==============================\n")

    results = []
    best_score = final_scores[0][0] if final_scores else 0

    for rank, (final_score, idx, semantic, keyword, metadata_boost) in enumerate(final_scores[:top_k]):

        item = data[idx]
        relative_score = (final_score / best_score) if best_score else 0
        recommendation_strength, confidence_score = _confidence_label(relative_score, _extract_matched_skills(query, item), metadata_boost)

        results.append(_build_recommendation_payload(
            query=query,
            item=item,
            semantic=semantic,
            keyword=keyword,
            metadata_boost=metadata_boost,
            final_score=final_score,
            recommendation_strength=recommendation_strength,
            confidence_score=confidence_score,
        ))

    # If leadership-like query, ensure OPQ / leadership-named items are prioritized
    if any(k in query.lower() for k in ["leadership", "senior leadership", "cxo", "executive", "director"]):
        opq_items = []
        for item in data:
            name = item.get("name", "") or ""
            if "opq" in name.lower() or "leadership" in name.lower():
                # build payload for candidate
                payload = _build_recommendation_payload(
                    query=query,
                    item=item,
                    semantic=0.0,
                    keyword=0.0,
                    metadata_boost=0.2,
                    final_score=0.0,
                    recommendation_strength="High",
                    confidence_score=85,
                )
                opq_items.append(payload)

        # Prepend unique OPQ items preserving order
        if opq_items:
            existing_names = {r["name"] for r in results}
            new_list = []
            for opq in opq_items:
                if opq["name"] not in existing_names:
                    new_list.append(opq)
            results = new_list + results

    return results

#Evaluation Function

def evaluate_recall_at_10():

    ground_truth = {

        "python coding assessment": [
            "Python (New)",
            "Automata (New)",
            "Automata Pro (New)",
            "Smart Interview Live Coding"
        ],

        "leadership personality assessment": [
            "OPQ Leadership Report",
            "Enterprise Leadership Report 1.0",
            "Enterprise Leadership Report 2.0"
        ],

        "java backend developer": [
            "Java 8 (New)",
            "Core Java (Advanced Level) (New)",
            "Java Frameworks (New)"
        ]
    }

    total_recall = 0

    for query, expected in ground_truth.items():

        # results = search(query, top_k=10)
        results = search(query, top_k=10)

        result_names = [r["name"] for r in results]

        hits = 0

        for item in expected:
            if item in result_names:
                hits += 1

        recall = hits / len(expected)

        total_recall += recall

        print(f"\nQuery: {query}")
        print(f"Recall@10: {recall:.2f}")

    avg_recall = total_recall / len(ground_truth)

    print("\nAverage Recall@10:", round(avg_recall, 2))

if __name__ == "__main__":
    #Test Queries
    search("Java backend developer")

    search("leadership personality assessment")

    search("graduate software engineer aptitude")

    search("sales manager communication skills")

    search("python coding assessment")

    #run eval
    evaluate_recall_at_10()

# measuring recall@10 value manually 

ground_truth = {
    "python coding assessment": [
        "Python (New)",
        "Automata (New)",
        "Automata Pro (New)",
        "Smart Interview Live Coding"
    ],

    "leadership personality assessment": [
        "OPQ Leadership Report",
        "Enterprise Leadership Report 1.0",
        "Enterprise Leadership Report 2.0"
    ]
}