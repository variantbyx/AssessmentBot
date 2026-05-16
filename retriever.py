import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

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


def get_model():
    global model
    if model is None:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("\nModel loaded successfully")
    return model


def load_or_build_index():
    if os.path.exists("shl_index.faiss"):
        loaded_index = faiss.read_index("shl_index.faiss")
        print("\nFAISS index loaded")
        print("Total vectors:", loaded_index.ntotal)
        return loaded_index

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

    # query_embedding = get_model().encode([query])
    query_embedding = get_model().encode([expanded_query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, len(data))

    semantic_scores = {}

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

        if "leadership" in query_lower and "Personality & Behavior" in item["keys"]:
            metadata_boost += 0.05

        if "aptitude" in query_lower and "Ability & Aptitude" in item["keys"]:
            metadata_boost += 0.05

        if "coding" in query_lower and "Simulations" in item["keys"]:
            metadata_boost += 0.03

        final_score = (
            0.75 * semantic +
            0.15 * keyword +
            metadata_boost
        )

        final_scores.append((final_score, idx))

    final_scores.sort(reverse=True)

    # print("\n==============================")
    # print("QUERY:", query)
    # print("==============================\n")

    results = []

    for rank, (_, idx) in enumerate(final_scores[:top_k]):

        item = data[idx]

        results.append({
            "name": item["name"],
            "url": item["link"],
            "categories": item["keys"],
            "job_levels": item["job_levels"]
        })

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