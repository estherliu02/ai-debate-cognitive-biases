MODEL_CONFIGS = {
    "planner": {
        "model": "openai/gpt-4.1-mini",
        "temperature": 0.4,
    },
    "speaker": {
        "model": "openai/gpt-4.1",
        "temperature": 0.7,
    },
    "evaluator": {
        "model": "openai/gpt-4.1-mini",
        "temperature": 0,
    },
    "suspect_evidence_analyzer": {
        "model": "openai/gpt-4.1",
        "temperature": 0,
    },
    "detective_bias_evaluators": [
        {"model": "openai/o3", "temperature": 0},
        {"model": "anthropic/claude-sonnet-4.6", "temperature": 0},
        {"model": "google/gemini-2.5-pro", "temperature": 0},
    ],
}
