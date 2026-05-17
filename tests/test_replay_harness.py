import os
import re
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("USE_SEMANTIC_SEARCH", "0")

from app import app


client = TestClient(app)

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces" / "GenAI_SampleConversations"


def _normalize_blockquote_text(block: str) -> str:
    lines = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            line = line[1:].lstrip()
        lines.append(line)
    return "\n".join(line for line in lines if line).strip()


def _extract_turn_blocks(md_text: str):
    turns = []
    matches = list(re.finditer(r"^### Turn\s+(\d+)\s*$", md_text, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(md_text)
        turns.append((int(match.group(1)), md_text[start:end]))
    return turns


def _extract_user_and_agent(turn_block: str):
    user_match = re.search(r"\*\*User\*\*\s*(.*?)\n\s*\*\*Agent\*\*", turn_block, flags=re.DOTALL)
    agent_match = re.search(r"\*\*Agent\*\*\s*(.*)$", turn_block, flags=re.DOTALL)
    if not user_match or not agent_match:
        raise AssertionError("Could not parse user/agent sections from trace turn")
    user_text = _normalize_blockquote_text(user_match.group(1))
    agent_text = agent_match.group(1).strip()
    return user_text, agent_text


def _extract_expected_recommendation_names(agent_text: str):
    table_lines = []
    in_table = False
    for raw_line in agent_text.splitlines():
        line = raw_line.strip()
        if line.startswith("| # | Name |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if re.match(r"^\|\s*\d+\s*\|", line):
                parts = [part.strip() for part in line.strip("|").split("|")]
                if len(parts) >= 2:
                    table_lines.append(parts[1])
    return table_lines


def _trace_end_flag(agent_text: str) -> bool:
    return "_`end_of_conversation`: **true**_" in agent_text.lower()


def _trace_has_recommendations(agent_text: str) -> bool:
    lower_text = agent_text.lower()
    return "no recommendations this turn" not in lower_text


def _assert_subsequence(expected, actual):
    actual_index = 0
    for item in expected:
        while actual_index < len(actual) and actual[actual_index] != item:
            actual_index += 1
        assert actual_index < len(actual), f"Expected recommendation '{item}' not found in response order"
        actual_index += 1


def _validate_response(resp_json):
    assert isinstance(resp_json.get("reply"), str)
    assert isinstance(resp_json.get("recommendations"), list)
    assert isinstance(resp_json.get("end_of_conversation"), bool)

    if resp_json.get("recommendations"):
        from retriever import data

        dataset_names = {item.get("name") for item in data}
        dataset_links = {item.get("link") for item in data if item.get("link")}
        for recommendation in resp_json["recommendations"]:
            assert (recommendation.get("name") in dataset_names) or (recommendation.get("url") in dataset_links)


def _load_md_traces():
    if not TRACE_DIR.is_dir():
        raise AssertionError(f"Missing official trace directory: {TRACE_DIR}")

    files = sorted(TRACE_DIR.glob("C*.md"))
    if len(files) != 10:
        raise AssertionError(f"Expected 10 trace markdown files in {TRACE_DIR}, found {len(files)}")

    traces = []
    for path in files:
        md_text = path.read_text(encoding="utf-8")
        turns = []
        for turn_number, turn_block in _extract_turn_blocks(md_text):
            user_text, agent_text = _extract_user_and_agent(turn_block)
            turns.append(
                {
                    "turn": turn_number,
                    "user": user_text,
                    "agent": agent_text,
                    "expected_names": _extract_expected_recommendation_names(agent_text),
                    "has_recommendations": _trace_has_recommendations(agent_text),
                    "end_of_conversation": _trace_end_flag(agent_text),
                }
            )
        if not turns:
            raise AssertionError(f"No turns parsed from {path.name}")
        traces.append({"name": path.name, "turns": turns})
    return traces


def test_replay_traces():
    traces = _load_md_traces()

    for trace in traces:
        history = []
        for turn in trace["turns"]:
            history.append({"role": "user", "content": turn["user"]})

            response = client.post("/chat", json={"messages": history})
            assert response.status_code == 200, trace["name"]
            body = response.json()
            _validate_response(body)

            recommendations = body["recommendations"]
            if recommendations:
                from retriever import data

                dataset_names = {item.get("name") for item in data}
                dataset_links = {item.get("link") for item in data if item.get("link")}
                for recommendation in recommendations:
                    assert (recommendation.get("name") in dataset_names) or (recommendation.get("url") in dataset_links), trace["name"]

            history.append({"role": "assistant", "content": turn["agent"]})
