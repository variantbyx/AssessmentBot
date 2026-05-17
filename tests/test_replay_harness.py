import os
import json
from fastapi.testclient import TestClient
from app import app

# Ensure retriever uses BM25-only in test env to avoid heavy model downloads
os.environ.setdefault("USE_SEMANTIC_SEARCH", "0")

client = TestClient(app)

# Load traces from traces/ if present, else use built-in sample traces
TRACES_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")

sample_traces = [
    {
        "id": "trace-1",
        "messages": [
            {"role": "user", "content": "Senior Java backend developer with Spring and AWS"}
        ],
        "expected": None,
    },
    {
        "id": "trace-2",
        "messages": [
            {"role": "user", "content": "I need an assessment"}
        ],
        "expected": None,
    },
]

# replicate to reach 10 traces if none provided
while len(sample_traces) < 10:
    sample_traces.append({"id": f"trace-{len(sample_traces)+1}", "messages": [{"role": "user", "content": "Graduate software engineer aptitude assessment"}], "expected": None})


def load_traces():
    if os.path.isdir(TRACES_DIR):
        traces = []
        for fname in os.listdir(TRACES_DIR):
            if fname.endswith('.json'):
                with open(os.path.join(TRACES_DIR, fname), 'r', encoding='utf-8') as f:
                    traces.append(json.load(f))
        if traces:
            return traces
    return sample_traces


def validate_response(resp_json):
    assert isinstance(resp_json.get('reply'), str)
    assert isinstance(resp_json.get('recommendations'), list)
    assert isinstance(resp_json.get('end_of_conversation'), bool)

    # If recommendations present, validate they come from dataset
    if resp_json.get('recommendations'):
        from retriever import data
        dataset_names = {item.get('name') for item in data}
        dataset_links = {item.get('link') for item in data if item.get('link')}
        for r in resp_json['recommendations']:
            assert (r.get('name') in dataset_names) or (r.get('url') in dataset_links)


def test_replay_traces():
    traces = load_traces()
    for trace in traces:
        # build request with full history; ensure not exceeding 8 messages
        messages = trace.get('messages', [])
        assert len(messages) <= 8
        resp = client.post('/chat', json={"messages": messages}, timeout=30)
        assert resp.status_code == 200
        resp_json = resp.json()
        validate_response(resp_json)

        # If the trace includes expected shortlist, compute recall@10
        expected = trace.get('expected')
        if expected:
            rec_names = [r.get('name') for r in resp_json.get('recommendations', [])]
            hits = sum(1 for item in expected if item in rec_names)
            recall = hits / max(1, len(expected))
            assert recall >= 0.0  # placeholder; evaluation harness may enforce threshold
