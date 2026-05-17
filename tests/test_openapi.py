import os
from fastapi.testclient import TestClient
from app import app

os.environ.setdefault("USE_SEMANTIC_SEARCH", "0")
client = TestClient(app)


def test_openapi_contains_chat_and_health():
    r = client.get('/openapi.json')
    assert r.status_code == 200
    doc = r.json()
    assert '/chat' in doc['paths']
    assert '/health' in doc['paths']

    # Ensure ChatResponse schema exists
    components = doc.get('components', {})
    schemas = components.get('schemas', {})
    assert 'ChatResponse' in schemas
    assert 'Recommendation' in schemas

    # Ensure examples for /chat request body exist
    chat_path = doc['paths']['/chat']
    post_op = chat_path.get('post', {})
    request_body = post_op.get('requestBody', {})
    content = request_body.get('content', {})
    assert 'application/json' in content
    app_json = content['application/json']
    examples = app_json.get('examples')
    assert examples is not None

    # Ensure responses contain examples
    responses = post_op.get('responses', {})
    ok_resp = responses.get('200', {})
    assert ok_resp, '200 response must be documented'
    ok_content = ok_resp.get('content', {})
    assert 'application/json' in ok_content
    ok_examples = ok_content['application/json'].get('examples')
    assert ok_examples is not None
