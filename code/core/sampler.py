from __future__ import annotations


class RejectionSampler:
    def __init__(self, max_attempts: int):
        self.max_attempts = max_attempts

    def should_accept(self, eval_result) -> bool:
        return bool(eval_result.passed)
