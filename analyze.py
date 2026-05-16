import json
from collections import Counter


with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total assessments:", len(data))

print("\nFields available:")
print(data[0].keys())

print("\nFirst assessment:\n")
print(json.dumps(data[0], indent=2))

missing_desc = 0

for item in data:
    if not item.get("description"):
        missing_desc += 1

print("\nMissing descriptions:", missing_desc)

counter = Counter()

for item in data:
    for k in item.get("keys", []):
        counter[k] += 1

print("\nTop categories:")
print(counter.most_common(10))

levels = set()

for item in data:
    for lvl in item.get("job_levels", []):
        levels.add(lvl)

print("\nJob levels:")
print(sorted(levels))

processed = []

for item in data:

    text = " ".join([
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", []))
    ])

    processed.append({
        "name": item.get("name"),
        "url": item.get("link"),
        "search_text": text
    })

print("\nExample processed item:\n")
print(json.dumps(processed[0], indent=2))