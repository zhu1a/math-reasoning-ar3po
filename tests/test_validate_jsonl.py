from tools.validate_jsonl import validate_record


def test_accepts_project_schemas():
    records = [
        {"question": "1+1?", "answer": "2", "source": "unit"},
        {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
        {
            "conversations": [{"from": "human", "value": "1+1?"}],
            "chosen": "2",
            "rejected": "3",
        },
    ]
    assert all(validate_record(record)[1] is None for record in records)


def test_rejects_malformed_and_equal_preferences():
    assert validate_record({"question": "", "answer": "2"})[1]
    assert validate_record({"conversations": []})[1]
    assert validate_record({
        "conversations": [{"from": "human", "value": "1+1?"}],
        "chosen": "2",
        "rejected": "2",
    })[1]
