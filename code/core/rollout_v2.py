from __future__ import annotations

import json
import re
from typing import Any

from configs.models import MODEL_CONFIGS
from configs import bias_runtime
from core.api_logging import DialogueApiLogger
from core.memory import AgentMemory, MemoryManager
from core.trait_evaluators import TRAIT_EVALUATOR_REGISTRY
from prompts.speaker_turn_detective import build_detective_speaker_turn_prompt
from schemas.dialogue_schema import Dialogue, DialogueTurn
from schemas.plan_schema import TransformationPlan
from utils.detective_claims import (
    accusation_clause_for_suspect,
    guilt_conclusion_for_suspect,
    innocence_conclusion_for_suspect,
    normalize_wrongdoing_event,
)
from utils.fact_units import observable_fact_unit, observable_fact_units, slot_relevant_fact_units
from utils.reason_text import normalize_anchor_reason
from utils.text_metrics import count_words

ADAPTIVE_REWRITE_TRAITS: set[str] = {"pro_jargon_bias"}
# DISABLED: old anchoring strategy menu removed. Anchoring now uses the
# selected incriminating reason from active_argument.reason_refs[0].
# ANCHORING_TECHNIQUE_NAMES = {
#     "evaluation_criterion",
#     "central_evidence",
#     "causal_story",
#     "burden_of_proof",
#     "reference_class",
#     "preemptive_weakness_framing",
#     "contrast_anchor",
#     "memorable_question_or_label",
# }
# DISABLED: confirmation_bias removed from the current experiment.
# CONFIRMATION_TECHNIQUE_NAMES = {
#     "hypothesis_activation",
#     "supporting_evidence_clustering",
#     "hypothesis_generated_predictions",
#     "counterevidence_assimilation",
#     "non_diagnostic_framing",
#     "selective_diagnosticity",
#     "preemptive_counterevidence_framing",
#     "belief_congruent_question",
#     "consistency_bridge",
#     "selective_recap",
# }
ANCHORING_TRANSFORMATION_TRAIT = "anchoring_bias"
# DISABLED: confirmation_bias removed from the current experiment.
# CONFIRMATION_TRANSFORMATION_TRAIT = "confirmation_bias"
FALLACY_TRAIT_PREFIX = "fallacy_trait__"
TRANSFORMATION_TARGET_TURN_TYPES = {"argument", "rebuttal", "summary"}
DETECTIVE_COMPACT_TURN_WORD_LIMITS = {
    "argument": {"target_min": 25, "target_max": 45, "hard_max": 55},
    "rebuttal": {"target_min": 25, "target_max": 45, "hard_max": 55},
    "summary": {"target_min": 25, "target_max": 45, "hard_max": 55},
}
PAIRWISE_CONTENT_PLAN_VERSION = "pairwise_role_slots_v2"
PAIRWISE_TURN_DEPENDENCIES = {
    1: [],
    2: [],
    3: [1],
    4: [2],
    5: [4],
    6: [3],
}
def _is_fallacy_trait_name(trait_name: str | None) -> bool:
    return isinstance(trait_name, str) and trait_name.startswith(FALLACY_TRAIT_PREFIX)


def _fallacy_subtype_from_trait_name(trait_name: str | None) -> str | None:
    if _is_fallacy_trait_name(trait_name):
        return str(trait_name).removeprefix(FALLACY_TRAIT_PREFIX)
    return None


class AdaptiveTurnGenerationError(RuntimeError):
    """Raised when an adaptive rewrite turn cannot satisfy validation after retries."""


class DebateRolloutEngine:
    def __init__(
        self,
        client,
        max_visible_turns: int = 4,
        use_state_summary: bool = True,
        max_turn_retries: int = 3,
        dialogue_api_logger: DialogueApiLogger | None = None,
    ):
        del use_state_summary
        self.client = client
        self.memory_manager = MemoryManager(client, max_visible_turns=max_visible_turns)
        self.max_turn_retries = max_turn_retries
        self.dialogue_api_logger = dialogue_api_logger
        self._dialogue_api_attempt_counts: dict[tuple[Any, ...], int] = {}
        self.last_opening_coverage: dict | None = None
        self.last_public_evidence_ids: set[int] | None = None
        self.last_public_evidence_bank: list[dict] | None = None

    def set_dialogue_api_logger(self, logger: DialogueApiLogger | None) -> None:
        self.dialogue_api_logger = logger

    def _dialogue_api_log_context(
        self,
        *,
        topic_cfg: dict,
        content_plan,
        turn_plan,
        trait_name: str,
        speaker_variant: str,
        speech_type: str,
        attempt: int,
        model: str,
        generation_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        setup = getattr(content_plan, "debate_setup", None) or topic_cfg.get("debate_setup") or {}
        speaker = getattr(turn_plan, "speaker", None)
        speaker_role_map = setup.get("speaker_role_map") or topic_cfg.get("speaker_role_map") or {}
        role = getattr(turn_plan, "speaker_role", None) or speaker_role_map.get(speaker)
        culprit_variant = setup.get("culprit_variant") or topic_cfg.get("culprit_variant")
        rival_variant = setup.get("rival_variant") or topic_cfg.get("rival_variant")
        condition_parts = [trait_name]
        if culprit_variant is not None:
            condition_parts.append(f"culprit-{culprit_variant}")
        if rival_variant is not None:
            condition_parts.append(f"rival-{rival_variant}")
        if len(condition_parts) == 1 and speaker_variant is not None:
            condition_parts.append(f"speaker-{speaker_variant}")
        return {
            "case_id": topic_cfg.get("case_id") or getattr(content_plan, "case_id", None),
            "condition": "__".join(str(part) for part in condition_parts if part),
            "ground_truth_order": topic_cfg.get("ground_truth_side_order"),
            "role": role,
            "speaker": speaker,
            "turn_id": getattr(turn_plan, "turn_id", None),
            "speech_type": speech_type,
            "attempt": attempt,
            "model": model,
            "generation_parameters": dict(generation_parameters),
        }

    @staticmethod
    def _request_snapshot_from_client(client, *, prompt: str, model: str, generation_parameters: dict[str, Any]) -> dict[str, Any]:
        payload = getattr(client, "last_json_request_payload", None)
        if isinstance(payload, dict):
            messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
            system_prompt = next(
                (message.get("content") for message in messages if isinstance(message, dict) and message.get("role") == "system"),
                None,
            )
            user_prompt = next(
                (message.get("content") for message in messages if isinstance(message, dict) and message.get("role") == "user"),
                prompt,
            )
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": messages,
            }
        return {
            "system_prompt": "You are a careful assistant that always returns valid JSON when asked.",
            "user_prompt": prompt,
            "messages": [
                {"role": "system", "content": "You are a careful assistant that always returns valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
        }

    @staticmethod
    def _request_snapshot_from_payload(payload: dict[str, Any] | None, *, prompt: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
            system_prompt = next(
                (message.get("content") for message in messages if isinstance(message, dict) and message.get("role") == "system"),
                None,
            )
            user_prompt = next(
                (message.get("content") for message in messages if isinstance(message, dict) and message.get("role") == "user"),
                prompt,
            )
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": messages,
            }
        return {
            "system_prompt": "You are a careful assistant that always returns valid JSON when asked.",
            "user_prompt": prompt,
            "messages": [
                {"role": "system", "content": "You are a careful assistant that always returns valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
        }

    @staticmethod
    def _response_snapshot_from_client(client, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw_text = getattr(client, "last_json_response_raw_text", None)
        parsed = payload if payload is not None else getattr(client, "last_json_parsed_output", None)
        if raw_text is None and payload is not None:
            raw_text = json.dumps(payload, ensure_ascii=False)
        return {
            "raw_text": raw_text,
            "parsed_output": parsed,
        }

    def _log_dialogue_api_attempt(
        self,
        *,
        context: dict[str, Any],
        prompt: str,
        payload: dict[str, Any] | None,
        status: str,
        error: str | None,
        next_attempt_number: int,
    ) -> int:
        if self.dialogue_api_logger is None:
            return 0
        attempt_key = (
            context.get("case_id"),
            context.get("condition"),
            context.get("ground_truth_order"),
            context.get("role"),
            context.get("speaker"),
            context.get("turn_id"),
            context.get("speech_type"),
        )
        next_attempt_number = self._dialogue_api_attempt_counts.get(attempt_key, 0) + 1
        attempt_records = getattr(self.client, "last_json_attempt_records", None)
        if isinstance(attempt_records, list) and attempt_records:
            for offset, attempt_record in enumerate(attempt_records):
                is_final_record = offset == len(attempt_records) - 1
                record_context = dict(context)
                record_context["attempt"] = next_attempt_number + offset
                record_status = status if is_final_record else attempt_record.get("status") or "error"
                record_error = error if is_final_record else attempt_record.get("error")
                self.dialogue_api_logger.log_attempt(
                    context=record_context,
                    request=self._request_snapshot_from_payload(
                        attempt_record.get("request_payload"),
                        prompt=prompt,
                    ),
                    response={
                        "raw_text": attempt_record.get("raw_text"),
                        "parsed_output": payload if is_final_record and payload is not None else attempt_record.get("parsed_output"),
                    },
                    status=record_status,
                    error=record_error,
                )
            self._dialogue_api_attempt_counts[attempt_key] = next_attempt_number + len(attempt_records) - 1
            return len(attempt_records)
        record_context = dict(context)
        record_context["attempt"] = next_attempt_number
        self.dialogue_api_logger.log_attempt(
            context=record_context,
            request=self._request_snapshot_from_client(
                self.client,
                prompt=prompt,
                model=context.get("model"),
                generation_parameters=context.get("generation_parameters") or {},
            ),
            response=self._response_snapshot_from_client(self.client, payload),
            status=status,
            error=error,
        )
        self._dialogue_api_attempt_counts[attempt_key] = next_attempt_number
        return 1

    @staticmethod
    def _evidence_ids_in_utterance(text: str | None) -> set[int]:
        return {int(match) for match in re.findall(r"\[E(\d+)\]", str(text or ""))}

    @staticmethod
    def _reported_evidence_ids(turn: DialogueTurn) -> set[int]:
        out: set[int] = set()
        for evidence_id in turn.evidence_citations or []:
            try:
                out.add(int(evidence_id))
            except (TypeError, ValueError):
                continue
        return out

    @classmethod
    def _all_turn_evidence_ids(cls, turn: DialogueTurn) -> set[int]:
        return cls._evidence_ids_in_utterance(turn.utterance) | cls._reported_evidence_ids(turn)

    @classmethod
    def _opening_public_evidence_ids(cls, turn_1: DialogueTurn, turn_2: DialogueTurn) -> set[int]:
        return cls._evidence_ids_in_utterance(turn_1.utterance) | cls._evidence_ids_in_utterance(turn_2.utterance)

    @staticmethod
    def _public_evidence_bank(evidence_bank: list[dict], public_evidence_ids: set[int]) -> list[dict]:
        public_ids = {int(evidence_id) for evidence_id in public_evidence_ids}
        return [
            item
            for item in evidence_bank
            if "index" in item and int(item["index"]) in public_ids
        ]

    @staticmethod
    def _is_pairwise_content_plan(content_plan) -> bool:
        return getattr(content_plan, "content_plan_version", None) == PAIRWISE_CONTENT_PLAN_VERSION

    @staticmethod
    def _reason_text_for_opening(reason: dict) -> list[str]:
        return [
            str(text).strip()
            for text in reason.get("evidence_text") or []
            if str(text).strip()
        ]

    @staticmethod
    def _reason_fact_units_for_opening(reason: dict) -> list[str]:
        return observable_fact_units(reason.get("fact_units") or [])

    @staticmethod
    def _projection_terms(texts: list[str]) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "could",
            "did", "does", "for", "from", "had", "has", "have", "he", "her", "him",
            "his", "i", "if", "in", "into", "is", "it", "its", "may", "more", "not",
            "of", "on", "or", "our", "she", "so", "than", "that", "the", "their",
            "them", "then", "there", "these", "they", "this", "to", "was", "we",
            "were", "which", "while", "who", "with", "would",
        }
        terms: set[str] = set()
        for text in texts:
            for token in re.findall(r"[A-Za-z0-9']+", str(text or "").lower()):
                normalized = token.strip("'")
                if len(normalized) < 3 or normalized in stopwords:
                    continue
                terms.add(normalized)
        return terms

    @staticmethod
    def _canonical_reason_statement(reason: dict) -> str:
        statements = reason.get("reason_statements")
        if isinstance(statements, list) and statements:
            first = statements[0]
            if isinstance(first, dict):
                return str(first.get("statement") or "")
        return ""

    @staticmethod
    def _reason_projection_text(reason: dict) -> str:
        return str(
            DebateRolloutEngine._canonical_reason_statement(reason)
            or reason.get("anchor_reason")
            or reason.get("reasoning")
            or reason.get("target_conclusion")
            or reason.get("conclusion")
            or ""
        ).strip()

    @staticmethod
    def _clean_projected_clause(clause: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(clause or "")).strip(" ,;")
        cleaned = re.sub(
            r"^(?:and|but|although|though|while|whereas|even though)\s+",
            "",
            cleaned,
            flags=re.I,
        ).strip(" ,;")
        return cleaned

    @staticmethod
    def _projection_clause_with_context(*, clause: str, reason: dict, previous_sentence: str | None) -> list[str]:
        suspect_name = str(reason.get("suspect_name") or "").strip()
        if suspect_name:
            clause = re.sub(r"^(?:he|she)\s+", f"{suspect_name} ", clause, count=1, flags=re.I)
        if re.match(r"^(it|they|them|this|that)\b", clause, flags=re.I) and previous_sentence:
            previous = DebateRolloutEngine._clean_projected_clause(previous_sentence)
            if previous and previous.casefold() != clause.casefold():
                return [previous, clause]
        return [clause]

    @classmethod
    def _reason_conditioned_evidence_projection(
        cls,
        *,
        reason: dict,
        source_text: str,
    ) -> str:
        source_text = str(source_text or "").strip()
        if not source_text:
            return ""
        reason_text = cls._reason_projection_text(reason)
        relevance_texts = [reason_text] if reason_text else [str(fact) for fact in (reason.get("fact_units") or [])]
        terms = cls._projection_terms(relevance_texts)
        if not terms:
            return source_text

        projected_clauses: list[str] = []
        minimum_overlap = 2 if len(terms) >= 3 else 1
        previous_sentence: str | None = None
        for sentence in re.split(r"(?<=[.!?])\s+", source_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            parts = [
                cls._clean_projected_clause(part)
                for part in re.split(
                    r";\s+|,\s+(?=(?:but|although|though|while|whereas|even though)\b)",
                    sentence,
                    flags=re.I,
                )
            ]
            parts = [part for part in parts if part]
            selected_parts = [
                part
                for part in parts
                if len(cls._projection_terms([part]) & terms) >= minimum_overlap
            ]
            if selected_parts:
                for part in selected_parts:
                    projected_clauses.extend(
                        cls._projection_clause_with_context(
                            clause=part,
                            reason=reason,
                            previous_sentence=previous_sentence,
                        )
                    )
            previous_sentence = sentence

        if not projected_clauses:
            return source_text

        deduped: list[str] = []
        seen: set[str] = set()
        for clause in projected_clauses:
            key = clause.casefold()
            if key in seen:
                continue
            seen.add(key)
            if clause and clause[-1] not in ".!?\"'":
                clause = f"{clause}."
            deduped.append(clause)
        return " ".join(deduped)

    @staticmethod
    def _selected_evidence_blocks_for_opening(
        *,
        reason_items: list[tuple[str, dict]],
        evidence_bank: list[dict] | None,
    ) -> list[dict]:
        selected: dict[int, dict] = {}
        fallback_order: list[int] = []
        fallback_text_by_id: dict[int, str] = {}
        for reason_key, reason in reason_items:
            evidence_ids = [int(evidence_id) for evidence_id in reason.get("evidence_ids") or []]
            evidence_texts = [
                str(text).strip()
                for text in reason.get("evidence_text") or []
            ]
            for offset, evidence_id in enumerate(evidence_ids):
                if evidence_id not in fallback_order:
                    fallback_order.append(evidence_id)
                if offset < len(evidence_texts) and evidence_id not in fallback_text_by_id:
                    fallback_text_by_id[evidence_id] = evidence_texts[offset]
                block = selected.setdefault(
                    evidence_id,
                    {
                        "evidence_id": evidence_id,
                        "reason_keys": [],
                        "reason_ids": [],
                        "directions": [],
                        "suspect_names": [],
                    },
                )
                if reason_key not in block["reason_keys"]:
                    block["reason_keys"].append(reason_key)
                reason_id = reason.get("reason_id")
                if reason_id and reason_id not in block["reason_ids"]:
                    block["reason_ids"].append(reason_id)
                direction = reason.get("direction")
                if direction and direction not in block["directions"]:
                    block["directions"].append(direction)
                suspect_name = reason.get("suspect_name")
                if suspect_name and suspect_name not in block["suspect_names"]:
                    block["suspect_names"].append(suspect_name)

        bank_text_by_id: dict[int, str] = {}
        bank_order: list[int] = []
        for item in evidence_bank or []:
            try:
                evidence_id = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if evidence_id in bank_text_by_id:
                continue
            bank_order.append(evidence_id)
            bank_text_by_id[evidence_id] = str(item.get("text") or "").strip()

        ordered_ids = [
            evidence_id
            for evidence_id in bank_order
            if evidence_id in selected
        ]
        ordered_ids.extend(
            evidence_id
            for evidence_id in fallback_order
            if evidence_id in selected and evidence_id not in set(ordered_ids)
        )

        blocks: list[dict] = []
        for evidence_id in ordered_ids:
            text = bank_text_by_id.get(evidence_id) or fallback_text_by_id.get(evidence_id) or ""
            block = dict(selected[evidence_id])
            block["evidence_text"] = text
            block["fact_units"] = observable_fact_units([text]) if text else []
            reason_projections: list[dict] = []
            projected_texts: list[str] = []
            seen_projected_texts: set[str] = set()
            for reason_key, reason in reason_items:
                reason_ids = [int(raw_id) for raw_id in reason.get("evidence_ids") or []]
                if evidence_id not in reason_ids:
                    continue
                projected_text = DebateRolloutEngine._reason_conditioned_evidence_projection(
                    reason=reason,
                    source_text=text,
                )
                projection = {
                    "reason_key": reason_key,
                    "reason_id": reason.get("reason_id"),
                    "suspect_name": reason.get("suspect_name"),
                    "direction": reason.get("direction"),
                    "projected_text": projected_text,
                }
                reason_projections.append(projection)
                projected_key = projected_text.casefold()
                if projected_text and projected_key not in seen_projected_texts:
                    seen_projected_texts.add(projected_key)
                    projected_texts.append(projected_text)
            block["reason_projections"] = reason_projections
            block["projected_evidence_text"] = " ".join(projected_texts) if projected_texts else text
            blocks.append(block)
        return blocks

    def _pairwise_opening_side_evidence(
        self,
        content_plan,
        *,
        speaker: str,
        evidence_bank: list[dict] | None = None,
    ) -> dict:
        selected = getattr(content_plan, "selected_reasons", None) or {}
        setup = getattr(content_plan, "debate_setup", None) or {}
        speaker_role_map = setup.get("speaker_role_map") or {}
        role = speaker_role_map.get(speaker)
        if role == "culprit":
            keys = ("culprit_incriminating", "rival_exculpatory")
            suspect_name = setup.get("culprit_name")
        elif role == "rival":
            keys = ("rival_incriminating", "culprit_exculpatory")
            suspect_name = setup.get("rival_suspect_name")
        else:
            raise AdaptiveTurnGenerationError(f"Unknown speaker for pairwise opening: {speaker!r}.")
        analysis_items: list[dict] = []
        reason_items: list[tuple[str, dict]] = []
        for key in keys:
            reason = selected.get(key)
            if not isinstance(reason, dict):
                raise AdaptiveTurnGenerationError(f"pairwise content plan missing selected reason {key}.")
            reason_items.append((key, reason))
            ids = [int(evidence_id) for evidence_id in reason.get("evidence_ids") or []]
            texts = self._reason_text_for_opening(reason)
            fact_units = self._reason_fact_units_for_opening(reason)
            analysis_items.append({
                "reason_key": key,
                "reason_id": reason.get("reason_id"),
                "suspect_name": reason.get("suspect_name"),
                "direction": reason.get("direction"),
                "evidence_indices": ids,
                "evidence_text": texts,
                "required_fact_units": fact_units,
            })
        evidence_blocks = self._selected_evidence_blocks_for_opening(
            reason_items=reason_items,
            evidence_bank=evidence_bank,
        )
        evidence_indices = [int(block["evidence_id"]) for block in evidence_blocks]
        evidence_text = [str(block.get("evidence_text") or "") for block in evidence_blocks]
        projected_evidence_text = [
            str(block.get("projected_evidence_text") or block.get("evidence_text") or "")
            for block in evidence_blocks
        ]
        return {
            "suspect_name": suspect_name,
            "side_role": "pairwise_side_a" if speaker == "A" else "pairwise_side_b",
            "evidence_indices": evidence_indices,
            "evidence_text": evidence_text,
            "projected_evidence_text": projected_evidence_text,
            "required_fact_units": projected_evidence_text,
            "evidence_blocks": evidence_blocks,
            "analysis_items": analysis_items,
        }

    @staticmethod
    def _topic_cfg_with_opening_side_evidence(topic_cfg: dict, side_evidence: dict) -> dict:
        scoped = dict(topic_cfg)
        # Keep the full story evidence bank available so openings can add
        # non-decisive background and transitions directly from the story.
        scoped["_opening_side_evidence"] = dict(side_evidence)
        return scoped

    @staticmethod
    def _opening_side_evidence_turn_plan(turn_plan, side_evidence: dict):
        return turn_plan.model_copy(update={
            "turn_type": "opening",
            "evidence_ids": list(side_evidence["evidence_indices"]),
            "fact_units": list(side_evidence.get("required_fact_units") or side_evidence["evidence_text"]),
            "case_theory": None,
            "narrative_beats": None,
            "claim_to_defend": None,
            "opponent_point_to_attack": None,
            "attack_type": None,
            "required_move": "Retell the full story using the supplied reason-conditioned evidence projections as the coverage checklist; use full evidence only as grounding support and add only non-decisive background from the original story for flow.",
            "allowed_concession": None,
            "end_state": "Opening factual reconstruction completed with all required evidence projections and clear story context.",
        })

    @staticmethod
    def _central_event_text(topic_cfg: dict) -> str:
        event = str(topic_cfg.get("wrongdoing_event") or "").strip()
        if event:
            return normalize_wrongdoing_event(event)
        question = str(topic_cfg.get("question") or topic_cfg.get("topic") or topic_cfg.get("motion") or "")
        match = re.search(r"\bresponsible\s+for\s+(.+?)\??$", question, flags=re.I)
        if match:
            return normalize_wrongdoing_event(match.group(1))
        return normalize_wrongdoing_event(question)

    @classmethod
    def _wrongdoing_event(cls, topic_cfg: dict, content_plan=None) -> str:
        setup = getattr(content_plan, "debate_setup", None) or topic_cfg.get("debate_setup") or {}
        bundle = cls._claim_realization_bundle(topic_cfg, content_plan)
        neutral_event = setup.get("neutral_event_description")
        if not neutral_event and isinstance(bundle, dict):
            neutral_event = bundle.get("neutral_event_description")
        if neutral_event:
            return normalize_wrongdoing_event(neutral_event)
        return normalize_wrongdoing_event(
            setup.get("wrongdoing_event")
            or topic_cfg.get("wrongdoing_event")
            or cls._central_event_text(topic_cfg)
        )

    @staticmethod
    def _claim_realization_bundle(topic_cfg: dict | None = None, content_plan=None) -> dict | None:
        setup = getattr(content_plan, "debate_setup", None) or (topic_cfg or {}).get("debate_setup") or {}
        bundle = getattr(content_plan, "claim_realization_bundle", None) or setup.get("claim_realization_bundle")
        return bundle if isinstance(bundle, dict) else None

    @classmethod
    def _accusation_clause(cls, *, suspect: str, topic_cfg: dict | None = None, content_plan=None) -> str:
        setup = getattr(content_plan, "debate_setup", None) or (topic_cfg or {}).get("debate_setup") or {}
        return accusation_clause_for_suspect(
            cls._claim_realization_bundle(topic_cfg, content_plan),
            suspect,
            setup.get("wrongdoing_event") or (topic_cfg or {}).get("wrongdoing_event"),
        )

    @classmethod
    def _guilt_conclusion(cls, *, suspect: str, topic_cfg: dict | None = None, content_plan=None) -> str:
        setup = getattr(content_plan, "debate_setup", None) or (topic_cfg or {}).get("debate_setup") or {}
        return guilt_conclusion_for_suspect(
            cls._claim_realization_bundle(topic_cfg, content_plan),
            suspect,
            setup.get("wrongdoing_event") or (topic_cfg or {}).get("wrongdoing_event"),
        )

    @classmethod
    def _innocence_conclusion(cls, *, suspect: str, topic_cfg: dict | None = None, content_plan=None) -> str:
        setup = getattr(content_plan, "debate_setup", None) or (topic_cfg or {}).get("debate_setup") or {}
        return innocence_conclusion_for_suspect(
            cls._claim_realization_bundle(topic_cfg, content_plan),
            suspect,
            setup.get("wrongdoing_event") or (topic_cfg or {}).get("wrongdoing_event"),
        )

    @staticmethod
    def _mask_nonpublic_evidence_references(text: str | None, public_evidence_ids: set[int] | None) -> str | None:
        if text is None or public_evidence_ids is None:
            return text
        public_ids = {int(evidence_id) for evidence_id in public_evidence_ids}
        referenced_ids = {int(match) for match in re.findall(r"\[E(\d+)\]", text)}
        if referenced_ids - public_ids:
            return (
                "Use only facts, claims, and evidence IDs already stated in Turn 1/2; "
                "ignore any original-plan references to non-public evidence."
            )

        def replace(match: re.Match) -> str:
            evidence_id = int(match.group(1))
            if evidence_id in public_ids:
                return match.group(0)
            return "[opening-public evidence only]"

        return re.sub(r"\[E(\d+)\]", replace, text)

    def _public_bounded_turn_plan(self, turn_plan, public_evidence_ids: set[int] | None):
        if public_evidence_ids is None:
            return turn_plan
        masked_fields = {
            field: self._mask_nonpublic_evidence_references(getattr(turn_plan, field), public_evidence_ids)
            for field in (
                "turn_goal",
                "claim_to_defend",
                "opponent_point_to_attack",
                "required_move",
                "allowed_concession",
                "end_state",
            )
        }
        return turn_plan.model_copy(update=masked_fields)

    @staticmethod
    def _has_semantic_contract(turn_plan) -> bool:
        return any(
            getattr(turn_plan, field, None) is not None
            for field in (
                "turn_type",
                "evidence_ids",
                "fact_units",
                "interpretation_units",
                "opponent_claim_target",
                "required_concession",
                "core_conclusion",
                "inference_edges",
                "certainty_level",
                "reason_refs",
                "sentence_jobs",
                "subject_suspect",
            )
        )

    @staticmethod
    def _evidence_ref_text(evidence_ids: list[int] | None) -> str:
        ids = [int(evidence_id) for evidence_id in (evidence_ids or [])]
        return " ".join(f"[E{evidence_id}]" for evidence_id in ids)

    @staticmethod
    def _join_units(values: list[str] | None) -> str:
        return "; ".join(str(value).strip() for value in (values or []) if str(value).strip())

    @staticmethod
    def _speaker_visible_fact_unit(text: str | None) -> str:
        return observable_fact_unit(text)

    @classmethod
    def _speaker_visible_fact_units(cls, values: list[str] | None) -> list[str]:
        return observable_fact_units(values)

    @staticmethod
    def _reason_index_from_selected_reasons(selected_reasons: dict | None) -> dict[str, dict]:
        return {
            str(reason.get("reason_id")): reason
            for reason in (selected_reasons or {}).values()
            if isinstance(reason, dict) and reason.get("reason_id")
        }

    def _verbosity_additional_reason_for_turn(
        self,
        *,
        turn_plan,
        reasons_by_id: dict[str, dict],
        trait_name: str | None,
        speaker_variant: str | None,
    ) -> dict | None:
        if trait_name != "verbosity_bias" or speaker_variant != "active":
            return None
        turn_type = getattr(turn_plan, "turn_type", None)
        if turn_type == "argument":
            expected_direction = "incriminating"
            local_purpose = "support_guilt"
            fact_sentence_index = 4
            reasoning_sentence_index = 5
            fact_purpose = "additional_incriminating_fact_sentence"
            reasoning_goal = "Explain why the additional incriminating fact separately supports guilt."
        elif turn_type == "rebuttal":
            expected_direction = "exculpatory"
            local_purpose = "support_innocence"
            fact_sentence_index = 6
            reasoning_sentence_index = 7
            fact_purpose = "additional_exculpatory_fact_sentence"
            reasoning_goal = "Explain why the additional exculpatory fact separately supports innocence."
        else:
            return None

        primary_reason = None
        for reason_id in getattr(turn_plan, "reason_refs", None) or []:
            candidate = reasons_by_id.get(str(reason_id))
            if isinstance(candidate, dict) and candidate.get("direction") == expected_direction:
                primary_reason = candidate
                break
        if not isinstance(primary_reason, dict):
            return None
        additional = primary_reason.get("additional_reason")
        if not isinstance(additional, dict):
            return None
        fact_units = slot_relevant_fact_units(
            additional.get("fact_units") or [],
            subject_suspect=getattr(turn_plan, "subject_suspect", None),
            local_purpose=local_purpose,
        )
        reasoning_guideline = normalize_anchor_reason(
            additional.get("anchor_reason")
            or self._canonical_reason_statement(additional)
            or additional.get("reasoning")
            or ""
        )
        return {
            "unit_id": "verbosity_additional_reason:1",
            "reason_id": additional.get("reason_id"),
            "parent_primary_reason_id": primary_reason.get("reason_id"),
            "direction": additional.get("direction") or expected_direction,
            "subject_suspect": getattr(turn_plan, "subject_suspect", None),
            "evidence_ids": list(additional.get("evidence_ids") or []),
            "fact_units": fact_units,
            "reasoning_guideline": reasoning_guideline,
            "fact_requirement": {
                "sentence_index": fact_sentence_index,
                "purpose": fact_purpose,
                "reason_id": additional.get("reason_id"),
                "direction": additional.get("direction") or expected_direction,
                "subject_suspect": getattr(turn_plan, "subject_suspect", None),
                "evidence_ids": list(additional.get("evidence_ids") or []),
                "fact_units": fact_units,
            },
            "reasoning_requirement": {
                "sentence_index": reasoning_sentence_index,
                "job": "fill_verbosity_additional_reasoning_slot",
                "slot_id": f"verbosity_additional_{expected_direction}_reasoning",
                "reason_id": additional.get("reason_id"),
                "local_purpose": local_purpose,
                "goal": reasoning_goal,
                "reasoning_guideline": reasoning_guideline,
            },
            "do_not_repeat_primary_reason": {
                "primary_reason_id": primary_reason.get("reason_id"),
                "primary_evidence_ids": list(primary_reason.get("evidence_ids") or []),
                "primary_fact_units": slot_relevant_fact_units(
                    primary_reason.get("fact_units") or [],
                    subject_suspect=getattr(turn_plan, "subject_suspect", None),
                    local_purpose=local_purpose,
                ),
            },
        }

    @staticmethod
    def _contract_visible_evidence_ids(locked_content_contract: dict | None) -> set[int] | None:
        if not isinstance(locked_content_contract, dict):
            return None
        ids: set[int] = set()
        for evidence_id in locked_content_contract.get("evidence_ids") or []:
            try:
                ids.add(int(evidence_id))
            except (TypeError, ValueError):
                continue
        for reason in locked_content_contract.get("referenced_reasons") or []:
            if not isinstance(reason, dict):
                continue
            for evidence_id in reason.get("evidence_ids") or []:
                try:
                    ids.add(int(evidence_id))
                except (TypeError, ValueError):
                    continue
        for requirement in locked_content_contract.get("fact_sentence_requirements") or []:
            if not isinstance(requirement, dict):
                continue
            for evidence_id in requirement.get("evidence_ids") or []:
                try:
                    ids.add(int(evidence_id))
                except (TypeError, ValueError):
                    continue
        additional = locked_content_contract.get("verbosity_additional_reason")
        if isinstance(additional, dict):
            for evidence_id in additional.get("evidence_ids") or []:
                try:
                    ids.add(int(evidence_id))
                except (TypeError, ValueError):
                    continue
        return ids

    @classmethod
    def _contract_scoped_evidence_bank(
        cls,
        evidence_bank: list[dict] | None,
        locked_content_contract: dict | None,
    ) -> list[dict] | None:
        visible_ids = cls._contract_visible_evidence_ids(locked_content_contract)
        if visible_ids is None or evidence_bank is None:
            return evidence_bank
        return [
            item
            for item in evidence_bank
            if "index" in item and int(item["index"]) in visible_ids
        ]

    def _pairwise_fact_sentence_requirements(
        self,
        *,
        turn_plan,
        reasons_by_id: dict[str, dict],
    ) -> list[dict]:
        def requirement_for_reason(*, sentence_index: int, purpose: str, reason: dict) -> dict:
            local_purpose = "support_guilt" if purpose == "incriminating_fact_sentence" else "support_innocence"
            return {
                "sentence_index": sentence_index,
                "purpose": purpose,
                "reason_id": reason.get("reason_id"),
                "direction": reason.get("direction"),
                "subject_suspect": turn_plan.subject_suspect,
                "evidence_ids": list(reason.get("evidence_ids") or []),
                "fact_units": slot_relevant_fact_units(
                    reason.get("fact_units") or [],
                    subject_suspect=turn_plan.subject_suspect,
                    local_purpose=local_purpose,
                ),
            }

        if getattr(turn_plan, "turn_type", None) == "argument":
            reason_id = (turn_plan.reason_refs or [None])[0]
            reason = reasons_by_id.get(str(reason_id))
            if isinstance(reason, dict):
                return [
                    requirement_for_reason(
                        sentence_index=2,
                        purpose="incriminating_fact_sentence",
                        reason=reason,
                    )
                ]
            return []

        if getattr(turn_plan, "turn_type", None) == "rebuttal":
            for reason_id in turn_plan.reason_refs or []:
                reason = reasons_by_id.get(str(reason_id))
                if isinstance(reason, dict) and reason.get("direction") == "exculpatory":
                    return [
                        requirement_for_reason(
                            sentence_index=4,
                            purpose="exculpatory_fact_sentence",
                            reason=reason,
                        )
                    ]
        return []

    def _semantic_contract_turn_plan(self, turn_plan, content_plan=None, topic_cfg: dict | None = None):
        if turn_plan.turn_id <= 2 or not self._has_semantic_contract(turn_plan):
            return turn_plan

        turn_type = turn_plan.turn_type or self._turn_type_for_id(turn_plan.turn_id)
        if turn_plan.sentence_jobs:
            evidence_refs = self._evidence_ref_text(turn_plan.evidence_ids)
            reason_refs = ", ".join(turn_plan.reason_refs or [])
            fact_text = self._join_units(self._speaker_visible_fact_units(turn_plan.fact_units))
            reasoning_slots = "; ".join(
                f"slot={job.get('slot_id')} "
                f"{'reason=' + str(job.get('reason_id')) if job.get('reason_id') else ''} "
                f"{job.get('goal') or ''}".strip()
                for job in turn_plan.sentence_jobs
                if isinstance(job, dict) and job.get("slot_id")
            )
            subject = str(turn_plan.subject_suspect or "the assigned suspect")
            core_conclusion = turn_plan.core_conclusion or "Preserve the assigned semantic conclusion."
            canonical_accusation = (
                turn_plan.accusation_clause
                or (
                    self._accusation_clause(
                        suspect=subject,
                        topic_cfg=topic_cfg or {},
                        content_plan=content_plan,
                    )
                    if subject != "the assigned suspect"
                    else None
                )
                or core_conclusion
            )
            if turn_type == "argument":
                goal = (
                    f"[ARGUMENT] use only fixed reason {reason_refs} and evidence {evidence_refs} "
                    f"to argue that {core_conclusion}"
                )
            else:
                goal = (
                    f"[REBUTTAL] use only fixed reason(s) {reason_refs} and evidence {evidence_refs} "
                    f"for the assigned insufficiency/exculpatory jobs about this conclusion: {core_conclusion}"
                )
            return turn_plan.model_copy(update={
                "turn_goal": goal,
                "claim_to_defend": f"{core_conclusion} {evidence_refs}".strip(),
                "opponent_point_to_attack": None,
                "attack_type": "rebuttal" if turn_type == "rebuttal" else None,
                "required_move": (
                    f"Use only these public fact units: {fact_text}. "
                    f"Use these semantic reasoning slots only: {reasoning_slots}. "
                    f"Use this stored canonical accusation without changing its semantic meaning: {canonical_accusation}. "
                    "The resolved dialogue structure, not the content plan, determines final sentence positions."
                ),
                "allowed_concession": None,
                "end_state": core_conclusion,
            })

        label = {
            "rebuttal": "REBUTTAL",
            "summary": "SUMMARY",
            "final_focus": "FINAL FOCUS",
        }.get(turn_type, str(turn_type or "TURN").upper())
        evidence_refs = self._evidence_ref_text(turn_plan.evidence_ids)
        fact_text = self._join_units(self._speaker_visible_fact_units(turn_plan.fact_units))
        interpretation_text = self._join_units(turn_plan.interpretation_units)
        edge_text = "; ".join(
            f"{', '.join(edge.source_fact_units)} -> {edge.interpretation_unit} -> {edge.supports_conclusion}"
            for edge in (turn_plan.inference_edges or [])
        )
        core_conclusion = turn_plan.core_conclusion or "Preserve the assigned semantic conclusion."
        update = {
            "turn_goal": f"[{label}] preserve this semantic conclusion using only the locked evidence ids {evidence_refs}: {core_conclusion}",
            "claim_to_defend": f"{core_conclusion} {evidence_refs}".strip(),
            "opponent_point_to_attack": turn_plan.opponent_claim_target,
            "attack_type": "rebuttal" if turn_type in {"rebuttal", "summary", "final_focus"} else None,
            "required_move": (
                f"Use only these fact units: {fact_text}. "
                f"Interpret them only as follows: {interpretation_text}. "
                f"Follow these inference edges: {edge_text}. "
                f"Certainty level: {turn_plan.certainty_level}."
            ),
            "allowed_concession": turn_plan.required_concession,
            "end_state": core_conclusion,
        }
        return turn_plan.model_copy(update=update)

    @staticmethod
    def _turn_plan_with_contract_evidence(turn_plan, locked_content_contract: dict | None):
        if not isinstance(locked_content_contract, dict):
            return turn_plan
        contract_evidence_ids = locked_content_contract.get("evidence_ids")
        if not isinstance(contract_evidence_ids, list):
            return turn_plan
        if list(getattr(turn_plan, "evidence_ids", None) or []) == contract_evidence_ids:
            return turn_plan
        return turn_plan.model_copy(update={"evidence_ids": list(contract_evidence_ids)})

    def _locked_content_contract(
        self,
        turn_plan,
        content_plan=None,
        *,
        trait_name: str | None = None,
        speaker_variant: str | None = None,
    ) -> dict | None:
        if turn_plan.turn_id <= 2 or not self._has_semantic_contract(turn_plan):
            return None
        if turn_plan.sentence_jobs:
            selected_reasons = getattr(content_plan, "selected_reasons", None) or {}
            reason_refs = list(turn_plan.reason_refs or [])
            reasons_by_id = self._reason_index_from_selected_reasons(selected_reasons)
            referenced_reason_objects = []
            for reason_id in reason_refs:
                reason = reasons_by_id.get(str(reason_id))
                if not isinstance(reason, dict):
                    continue
                referenced_reason_objects.append({
                    "reason_id": reason.get("reason_id"),
                    "suspect_name": reason.get("suspect_name"),
                    "direction": reason.get("direction"),
                    "evidence_ids": list(reason.get("evidence_ids") or []),
                    "fact_units": slot_relevant_fact_units(
                        reason.get("fact_units") or [],
                        subject_suspect=turn_plan.subject_suspect,
                        local_purpose=(
                            "support_guilt"
                            if reason.get("direction") == "incriminating"
                            else "support_innocence"
                        ),
                    ),
                    "target_conclusion": reason.get("target_conclusion") or reason.get("conclusion"),
                })
            fact_units = [
                {"unit_id": f"fact:{idx}", "text": text}
                for idx, text in enumerate(self._speaker_visible_fact_units(turn_plan.fact_units), start=1)
            ]
            sentence_jobs = []
            for idx, job in enumerate(turn_plan.sentence_jobs or [], start=1):
                if not isinstance(job, dict):
                    continue
                unit_id = str(job.get("slot_id") or f"sentence_job:{idx}")
                sentence_jobs.append({
                    "unit_id": unit_id,
                    **job,
                })
            subject = str(turn_plan.subject_suspect or "the assigned suspect")
            wrongdoing_event = self._wrongdoing_event({}, content_plan)
            compact_conclusion = turn_plan.core_conclusion or "Preserve the assigned semantic conclusion."
            canonical_accusation = (
                turn_plan.accusation_clause
                or (
                    self._accusation_clause(
                        suspect=subject,
                        content_plan=content_plan,
                    )
                    if subject != "the assigned suspect"
                    else None
                )
                or compact_conclusion
            )
            fact_sentence_requirements = self._pairwise_fact_sentence_requirements(
                turn_plan=turn_plan,
                reasons_by_id=reasons_by_id,
            )
            additional_reason = self._verbosity_additional_reason_for_turn(
                turn_plan=turn_plan,
                reasons_by_id=reasons_by_id,
                trait_name=trait_name,
                speaker_variant=speaker_variant,
            )
            if additional_reason is not None:
                fact_sentence_requirements.append(dict(additional_reason["fact_requirement"]))
                sentence_jobs.append({
                    "unit_id": "verbosity_additional_reasoning:1",
                    **additional_reason["reasoning_requirement"],
                })
                turn_plan_evidence_ids = list(turn_plan.evidence_ids or [])
                seen_evidence_ids = {int(evidence_id) for evidence_id in turn_plan_evidence_ids}
                for evidence_id in additional_reason.get("evidence_ids") or []:
                    evidence_id = int(evidence_id)
                    if evidence_id not in seen_evidence_ids:
                        seen_evidence_ids.add(evidence_id)
                        turn_plan_evidence_ids.append(evidence_id)
            else:
                turn_plan_evidence_ids = list(turn_plan.evidence_ids or [])
            return {
                "content_plan_version": PAIRWISE_CONTENT_PLAN_VERSION,
                "wrongdoing_event": wrongdoing_event,
                "turn_id": turn_plan.turn_id,
                "turn_type": turn_plan.turn_type,
                "speaker": turn_plan.speaker,
                "subject_suspect": turn_plan.subject_suspect,
                "canonical_accusation": {
                    "unit_id": "canonical_accusation:1",
                    "text": canonical_accusation,
                },
                "opponent_claim_target": (
                    {
                        "unit_id": "opponent_claim_target:1",
                        "text": turn_plan.opponent_claim_target,
                    }
                    if turn_plan.opponent_claim_target
                    else None
                ),
                "reason_refs": reason_refs,
                "referenced_reasons": referenced_reason_objects,
                "sentence_jobs": sentence_jobs,
                "evidence_ids": turn_plan_evidence_ids,
                "fact_units": fact_units,
                "fact_sentence_requirements": fact_sentence_requirements,
                **({"verbosity_additional_reason": additional_reason} if additional_reason is not None else {}),
                "core_conclusion": {
                    "unit_id": "core_conclusion:1",
                    "text": compact_conclusion,
                },
                "allowed_variation": [
                    "reasoning_slot_text",
                    "reason-to-guilt connection",
                    "insufficiency explanation when assigned",
                    "opponent-reason framing when assigned",
                ],
            }
        fact_units = [
            {"unit_id": f"fact:{idx}", "text": text}
            for idx, text in enumerate(self._speaker_visible_fact_units(turn_plan.fact_units), start=1)
        ]
        interpretation_units = [
            {"unit_id": f"interpretation:{idx}", "text": text}
            for idx, text in enumerate(turn_plan.interpretation_units or [], start=1)
        ]
        inference_edges = [
            {
                "unit_id": f"inference_edge:{idx}",
                "source_fact_units": list(edge.source_fact_units),
                "interpretation_unit": edge.interpretation_unit,
                "supports_conclusion": edge.supports_conclusion,
            }
            for idx, edge in enumerate(turn_plan.inference_edges or [], start=1)
        ]
        contract = {
            "turn_id": turn_plan.turn_id,
            "turn_type": turn_plan.turn_type,
            "speaker": turn_plan.speaker,
            "evidence_ids": list(turn_plan.evidence_ids or []),
            "fact_units": fact_units,
            "interpretation_units": interpretation_units,
            "opponent_claim_target": {
                "unit_id": "opponent_claim_target:1",
                "text": turn_plan.opponent_claim_target,
            },
            "required_concession": (
                {"unit_id": "concession:1", "text": turn_plan.required_concession}
                if turn_plan.required_concession
                else None
            ),
            "core_conclusion": {
                "unit_id": "core_conclusion:1",
                "text": turn_plan.core_conclusion,
            },
            "inference_edges": inference_edges,
            "certainty_level": turn_plan.certainty_level,
        }
        return contract

    @staticmethod
    def _contract_unit_ids(locked_content_contract: dict | None) -> set[str]:
        if not locked_content_contract:
            return set()
        unit_ids = {
            item["unit_id"]
            for field in ("fact_units", "interpretation_units", "inference_edges", "sentence_jobs")
            for item in locked_content_contract.get(field, []) or []
            if isinstance(item, dict) and item.get("unit_id")
        }
        for field in ("canonical_accusation", "opponent_claim_target", "required_concession", "core_conclusion"):
            item = locked_content_contract.get(field)
            if isinstance(item, dict) and item.get("unit_id"):
                unit_ids.add(item["unit_id"])
        return unit_ids

    @staticmethod
    def _default_ordered_unit_ids(locked_content_contract: dict | None) -> list[str]:
        if not locked_content_contract:
            return []
        if locked_content_contract.get("content_plan_version") == PAIRWISE_CONTENT_PLAN_VERSION:
            out: list[str] = []
            out.extend(item["unit_id"] for item in locked_content_contract.get("fact_units", []) if item.get("unit_id"))
            accusation = locked_content_contract.get("canonical_accusation")
            if isinstance(accusation, dict) and accusation.get("unit_id"):
                out.append(accusation["unit_id"])
            out.extend(item["unit_id"] for item in locked_content_contract.get("sentence_jobs", []) if item.get("unit_id"))
            conclusion = locked_content_contract.get("core_conclusion")
            if isinstance(conclusion, dict) and conclusion.get("unit_id"):
                out.append(conclusion["unit_id"])
            return out
        out: list[str] = []
        target = locked_content_contract.get("opponent_claim_target")
        if isinstance(target, dict) and target.get("unit_id"):
            out.append(target["unit_id"])
        out.extend(item["unit_id"] for item in locked_content_contract.get("fact_units", []) if item.get("unit_id"))
        concession = locked_content_contract.get("required_concession")
        if isinstance(concession, dict) and concession.get("unit_id"):
            out.append(concession["unit_id"])
        out.extend(item["unit_id"] for item in locked_content_contract.get("interpretation_units", []) if item.get("unit_id"))
        out.extend(item["unit_id"] for item in locked_content_contract.get("inference_edges", []) if item.get("unit_id"))
        conclusion = locked_content_contract.get("core_conclusion")
        if isinstance(conclusion, dict) and conclusion.get("unit_id"):
            out.append(conclusion["unit_id"])
        return out

    def _transformation_plan_for_turn(
        self,
        *,
        style_bundle: dict,
        speaker: str,
        turn_type: str,
        locked_content_contract: dict | None,
        anchor_spec: dict | None = None,
    ) -> TransformationPlan | None:
        raw_plan = style_bundle.get(speaker) if style_bundle else None
        if raw_plan is None or locked_content_contract is None:
            return None
        if isinstance(raw_plan, TransformationPlan):
            plan = raw_plan
        elif hasattr(raw_plan, "model_dump") and "allowed_surface_operations" in raw_plan.model_dump():
            plan = TransformationPlan(**raw_plan.model_dump())
        elif hasattr(raw_plan, "allowed_surface_operations"):
            plan = TransformationPlan(
                trait_name=getattr(raw_plan, "trait_name", "unknown"),
                variant=getattr(raw_plan, "variant", "baseline"),
                anchor_unit_id=getattr(raw_plan, "anchor_unit_id", None),
                provisional_side=getattr(raw_plan, "provisional_side", None),
                fallacy_subtype=getattr(raw_plan, "fallacy_subtype", None),
                hook_unit_ids=getattr(raw_plan, "hook_unit_ids", None),
                branch_metadata=getattr(raw_plan, "branch_metadata", None),
                ordered_unit_ids=list(getattr(raw_plan, "ordered_unit_ids", [])),
                foreground_unit_ids=list(getattr(raw_plan, "foreground_unit_ids", [])),
                background_unit_ids=list(getattr(raw_plan, "background_unit_ids", [])),
                repeat_unit_ids=list(getattr(raw_plan, "repeat_unit_ids", [])),
                allowed_surface_operations=list(getattr(raw_plan, "allowed_surface_operations", [])),
                allowed_inference_operations=list(getattr(raw_plan, "allowed_inference_operations", [])),
            )
        else:
            plan = TransformationPlan(
                trait_name="legacy",
                variant="identity",
                anchor_unit_id=None,
                provisional_side=None,
                fallacy_subtype=None,
                hook_unit_ids=None,
                branch_metadata=None,
                ordered_unit_ids=[],
                foreground_unit_ids=[],
                background_unit_ids=[],
                repeat_unit_ids=[],
                allowed_surface_operations=["identity rendering"],
                allowed_inference_operations=["preserve locked inference_edges exactly"],
            )
        if turn_type not in {"argument", "rebuttal"}:
            return None
        valid_ids = self._contract_unit_ids(locked_content_contract)
        ordered_unit_ids = plan.ordered_unit_ids or self._default_ordered_unit_ids(locked_content_contract)
        update = {"ordered_unit_ids": ordered_unit_ids}
        metadata = dict(plan.branch_metadata or {})
        slot_realizations = metadata.get("reasoning_slot_realizations") or {}
        selected_slot_realizations: dict[str, dict] = {}
        if isinstance(slot_realizations, dict):
            subtype_for_slots = plan.fallacy_subtype or _fallacy_subtype_from_trait_name(plan.trait_name)
            bias_key = subtype_for_slots if subtype_for_slots is not None else plan.trait_name
            for item in locked_content_contract.get("sentence_jobs", []) or []:
                if not isinstance(item, dict) or not item.get("slot_id"):
                    continue
                slot_id = str(item["slot_id"])
                realization = slot_realizations.get(slot_id)
                if not isinstance(realization, dict):
                    continue
                if plan.variant == "active" and plan.trait_name != "pro_jargon_bias":
                    biased_guidelines = realization.get("biased_guideline") or {}
                    selected_text = biased_guidelines.get(bias_key) if isinstance(biased_guidelines, dict) else None
                else:
                    selected_text = realization.get("baseline_guideline")
                if selected_text:
                    selected_slot_realizations[slot_id] = {
                        "slot_id": slot_id,
                        "local_purpose": realization.get("local_purpose"),
                        "sentence_index": item.get("sentence_index"),
                        "job": item.get("job"),
                        "subject_suspect": realization.get("subject_suspect"),
                        "claim_target": realization.get("claim_target"),
                        "fact_units": list(realization.get("fact_units") or []),
                        "required_conclusion": realization.get("required_conclusion") or realization.get("core_conclusion"),
                        "guideline": selected_text,
                    }
            if selected_slot_realizations:
                metadata["selected_reasoning_slot_realizations"] = selected_slot_realizations
                update["branch_metadata"] = metadata
        if plan.trait_name == ANCHORING_TRANSFORMATION_TRAIT and plan.variant == "active":
            if isinstance(anchor_spec, dict) and anchor_spec:
                metadata = dict(update.get("branch_metadata") or plan.branch_metadata or {})
                metadata["anchor_spec"] = dict(anchor_spec)
                update["branch_metadata"] = metadata
                update["anchor_unit_id"] = None
                update["repeat_unit_ids"] = []
        # DISABLED: confirmation_bias removed from the current experiment.
        # if plan.trait_name == CONFIRMATION_TRANSFORMATION_TRAIT and plan.variant == "active":
        #     selected_side = self._normalize_provisional_side(
        #         provisional_side or plan.provisional_side or speaker
        #     )
        #     foreground_ids, background_ids = self._confirmation_unit_groups(
        #         locked_content_contract=locked_content_contract,
        #         provisional_side=selected_side,
        #     )
        #     ordered_unit_ids = [
        #         *foreground_ids,
        #         *[
        #             unit_id
        #             for unit_id in ordered_unit_ids
        #             if unit_id not in set(foreground_ids) | set(background_ids)
        #         ],
        #         *background_ids,
        #     ]
        #     branch_metadata = dict(update.get("branch_metadata") or plan.branch_metadata or {})
        #     branches = dict(branch_metadata.get("branches") or {})
        #     branch = dict(branches.get(selected_side) or {})
        #     branch["provisional_side"] = selected_side
        #     branches[selected_side] = branch
        #     branch_metadata["branches"] = branches
        #     branch_metadata["selected_provisional_side"] = selected_side
        #     update["provisional_side"] = selected_side
        #     update["branch_metadata"] = branch_metadata
        #     update["ordered_unit_ids"] = ordered_unit_ids
        #     update["foreground_unit_ids"] = foreground_ids
        #     update["background_unit_ids"] = background_ids
        if not plan.foreground_unit_ids and "foreground_unit_ids" not in update:
            conclusion = locked_content_contract.get("core_conclusion") or {}
            if conclusion.get("unit_id"):
                update["foreground_unit_ids"] = [conclusion["unit_id"]]
        bound_plan = plan.model_copy(update=update)
        return bound_plan

    @staticmethod
    def _select_anchor_unit_id(locked_content_contract: dict | None) -> str | None:
        if not locked_content_contract:
            return None
        for item in locked_content_contract.get("interpretation_units", []) or []:
            if isinstance(item, dict) and item.get("unit_id"):
                return str(item["unit_id"])
        return None

    def _selected_reason_by_id(self, content_plan) -> dict[str, dict]:
        return {
            str(reason.get("reason_id")): dict(reason)
            for reason in (getattr(content_plan, "selected_reasons", None) or {}).values()
            if isinstance(reason, dict) and reason.get("reason_id")
        }

    def _select_dialogue_level_anchor(
        self,
        *,
        content_plan,
        active_speaker: str,
        public_evidence_ids: set[int] | None,
    ) -> dict | None:
        del public_evidence_ids
        active_argument = next(
            (
                turn_plan
                for turn_plan in content_plan.turns
                if turn_plan.speaker == active_speaker
                and getattr(turn_plan, "turn_type", None) == "argument"
                and getattr(turn_plan, "reason_refs", None)
            ),
            None,
        )
        if active_argument is None:
            raise AdaptiveTurnGenerationError(
                f"Anchoring requires an active first argument turn for speaker {active_speaker!r} with reason_refs."
            )
        source_reason_id = str(active_argument.reason_refs[0])
        reason = self._selected_reason_by_id(content_plan).get(source_reason_id)
        if not isinstance(reason, dict):
            raise AdaptiveTurnGenerationError(
                f"Anchoring source_reason_id {source_reason_id!r} was not found in content_plan.selected_reasons."
            )
        if reason.get("direction") != "incriminating":
            raise AdaptiveTurnGenerationError(
                f"Anchoring source_reason_id {source_reason_id!r} must be the active argument's incriminating reason."
            )
        anchor_reason = normalize_anchor_reason(
            reason.get("anchor_reason")
            or reason.get("reasoning")
            or self._canonical_reason_statement(reason)
            or reason.get("target_conclusion")
            or reason.get("conclusion")
            or ""
        ).strip()
        if not anchor_reason:
            raise AdaptiveTurnGenerationError(
                f"Anchoring source_reason_id {source_reason_id!r} is missing a complete incriminating reasoning claim."
            )
        return {
            "source_reason_id": source_reason_id,
            "anchor_reason": anchor_reason,
            "supporting_fact_units": list(reason.get("fact_units") or []),
            "supporting_evidence_ids": list(reason.get("evidence_ids") or []),
        }

    def _dialogue_level_anchor_spec_for_rollout(
        self,
        *,
        content_plan,
        trait_name: str,
        speaker_variants: dict[str, str],
    ) -> dict | None:
        if trait_name != ANCHORING_TRANSFORMATION_TRAIT:
            return None
        active_speaker = next(
            (speaker for speaker, variant in speaker_variants.items() if variant == "active"),
            None,
        )
        if active_speaker is None:
            return None
        return self._select_dialogue_level_anchor(
            content_plan=content_plan,
            active_speaker=active_speaker,
            public_evidence_ids=None,
        )

    # DISABLED: confirmation_bias removed from the current experiment.
    # @staticmethod
    # def _normalize_provisional_side(value: str | None) -> str:
    #     side = str(value or "").strip().upper()
    #     if side not in {"A", "B"}:
    #         raise AdaptiveTurnGenerationError(
    #             f"confirmation_bias provisional_side must be 'A' or 'B', got {value!r}."
    #         )
    #     return side
    #
    # def _confirmation_unit_groups(
    #     self,
    #     *,
    #     locked_content_contract: dict | None,
    #     provisional_side: str,
    # ) -> tuple[list[str], list[str]]:
    #     if not locked_content_contract:
    #         return [], []
    #     contract_speaker = str(locked_content_contract.get("speaker") or "").strip().upper()
    #     interpretation_ids = [
    #         item["unit_id"]
    #         for item in locked_content_contract.get("interpretation_units", []) or []
    #         if isinstance(item, dict) and item.get("unit_id")
    #     ]
    #     conclusion = locked_content_contract.get("core_conclusion")
    #     conclusion_ids = [conclusion["unit_id"]] if isinstance(conclusion, dict) and conclusion.get("unit_id") else []
    #     target = locked_content_contract.get("opponent_claim_target")
    #     target_ids = [target["unit_id"]] if isinstance(target, dict) and target.get("unit_id") else []
    #     concession = locked_content_contract.get("required_concession")
    #     concession_ids = [concession["unit_id"]] if isinstance(concession, dict) and concession.get("unit_id") else []
    #     if provisional_side == contract_speaker:
    #         foreground_ids = [*interpretation_ids, *conclusion_ids]
    #         background_ids = [*target_ids, *concession_ids]
    #     else:
    #         foreground_ids = [*target_ids, *concession_ids]
    #         background_ids = [*interpretation_ids, *conclusion_ids]
    #     return foreground_ids, background_ids

    @staticmethod
    def _turn_type_for_id(turn_id: int) -> str:
        if turn_id in {3, 4}:
            return "rebuttal"
        if turn_id in {5, 6}:
            return "summary"
        if turn_id in {7, 8}:
            return "final_focus"
        return "opening"

    @staticmethod
    def _effective_turn_type(turn_plan) -> str:
        return getattr(turn_plan, "turn_type", None) or DebateRolloutEngine._turn_type_for_id(turn_plan.turn_id)

    @staticmethod
    def _topic_cfg_with_public_evidence(topic_cfg: dict, public_evidence_bank: list[dict] | None) -> dict:
        if public_evidence_bank is None:
            return topic_cfg
        bounded_topic_cfg = dict(topic_cfg)
        bounded_topic_cfg["evidence_bank"] = list(public_evidence_bank)
        return bounded_topic_cfg

    @staticmethod
    def _sentence_candidates(text: str | None) -> list[str]:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return []
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    # DISABLED: confirmation_bias removed from the current experiment.
    # def _is_shallow_confirmation_hypothesis(...): ...
    # def _substantive_confirmation_hypothesis(...): ...

    # DISABLED: old anchoring strategy menu removed from runtime.
    # def _select_anchoring_techniques(...): ...
    #
    # DISABLED: confirmation_bias removed from the current experiment.
    # def _select_confirmation_techniques(...): ...

    def _adaptive_treatment_metadata(
        self,
        *,
        trait_name: str,
        speaker_variants: dict[str, str],
        fixed_dialogue: Dialogue | None,
        topic_cfg: dict | None = None,
    ) -> dict:
        del trait_name, speaker_variants, fixed_dialogue, topic_cfg
        return {}

    @staticmethod
    def _adaptive_forbidden_patterns(trait_name: str) -> list[str]:
        del trait_name
        return []

    @staticmethod
    def _adaptive_speech_instruction(trait_name: str, speech_type: str) -> str:
        del speech_type
        if trait_name == "pro_jargon_bias":
            return "Use the pro_jargon_bias whole-turn rewrite prompt."
        return "Preserve baseline substance while changing presentation strategy."

    def _adaptive_rewrite_context(
        self,
        *,
        trait_name: str,
        turn_plan,
        speech_type: str,
        fixed_turn_map: dict[int, DialogueTurn],
        treatment_metadata: dict | None,
        public_evidence_ids: set[int] | None = None,
    ) -> dict | None:
        if trait_name not in ADAPTIVE_REWRITE_TRAITS:
            return None
        if turn_plan.turn_id <= 2 or speech_type == "final_focus":
            return None
        baseline_turn = fixed_turn_map.get(turn_plan.turn_id)
        if baseline_turn is None:
            return None
        metadata = treatment_metadata or {}
        return {
            "adaptive_treatment": trait_name,
            "baseline_source_utterance": baseline_turn.utterance,
            "baseline_evidence_citations": list(baseline_turn.evidence_citations or []),
            "baseline_claims": [
                turn_plan.claim_to_defend,
                turn_plan.required_move,
                turn_plan.end_state,
            ],
            "baseline_opponent_claim_addressed": baseline_turn.opponent_claim_targeted or turn_plan.opponent_point_to_attack,
            "main_opposing_challenge": metadata.get("main_opposing_challenge"),
            # DISABLED: confirmation_bias removed from the current experiment.
            # "hypothesis_spec": metadata.get("hypothesis_spec"),
            # "confirmation_techniques": metadata.get("confirmation_techniques"),
            # "assumed_judge_provisional_side": metadata.get("assumed_judge_provisional_side"),
            # "active_side_matches_provisional_choice": metadata.get("active_side_matches_provisional_choice"),
            "continuation_turn_index": self._speaker_post_opening_index(turn_plan.turn_id, turn_plan.speaker),
            "speech_instruction": self._adaptive_speech_instruction(trait_name, speech_type),
            "forbidden_patterns": self._adaptive_forbidden_patterns(trait_name),
        }

    @staticmethod
    def _speaker_post_opening_index(turn_id: int, speaker: str) -> int:
        if speaker == "A":
            return max(0, (turn_id - 1) // 2)
        if speaker == "B":
            return max(0, (turn_id - 2) // 2)
        return 0

    def _build_memories(self, topic_cfg: dict, style_bundle: dict) -> dict[str, AgentMemory]:
        return {
            "A": AgentMemory(
                agent_name="A",
                stance=topic_cfg["agent_a_stance"],
                style_rules=list(getattr(style_bundle.get("A"), "behavioral_rules", [])),
            ),
            "B": AgentMemory(
                agent_name="B",
                stance=topic_cfg["agent_b_stance"],
                style_rules=list(getattr(style_bundle.get("B"), "behavioral_rules", [])),
            ),
        }

    @staticmethod
    def _topic_cfg_for_content_plan(topic_cfg: dict, content_plan) -> dict:
        if getattr(content_plan, "content_plan_version", None) != PAIRWISE_CONTENT_PLAN_VERSION:
            return topic_cfg
        setup = getattr(content_plan, "debate_setup", None) or {}
        scoped = dict(topic_cfg)
        scoped.update({
            "content_plan_version": PAIRWISE_CONTENT_PLAN_VERSION,
            "topic": getattr(content_plan, "topic", None) or setup.get("question") or topic_cfg.get("topic"),
            "question": getattr(content_plan, "debate_question", None) or setup.get("question") or topic_cfg.get("question"),
            "motion": getattr(content_plan, "debate_question", None) or setup.get("question") or topic_cfg.get("motion"),
            "agent_a_stance": getattr(content_plan, "agent_a_stance", None) or setup.get("agent_a_stance") or topic_cfg.get("agent_a_stance"),
            "agent_b_stance": getattr(content_plan, "agent_b_stance", None) or setup.get("agent_b_stance") or topic_cfg.get("agent_b_stance"),
            "wrongdoing_event": DebateRolloutEngine._wrongdoing_event(topic_cfg, content_plan),
            "debate_setup": setup,
        })
        return scoped

    def _speech_type_settings(self, turn_plan, total_turns: int, topic_cfg: dict) -> tuple[str, int]:
        semantic_turn_type = getattr(turn_plan, "turn_type", None)
        is_pairwise_v2 = topic_cfg.get("content_plan_version") == PAIRWISE_CONTENT_PLAN_VERSION
        turn_goal_upper = (turn_plan.turn_goal or "").upper()
        if is_pairwise_v2:
            if semantic_turn_type not in {"opening", "argument", "rebuttal"}:
                raise AdaptiveTurnGenerationError(
                    f"Turn {turn_plan.turn_id} pairwise_role_slots_v2 turn_type must be "
                    "'opening', 'argument', or 'rebuttal'."
                )
            speech_type = semantic_turn_type
            if speech_type == "opening":
                max_tokens = 1100
            else:
                max_tokens = 950
        elif turn_plan.turn_id <= 2:
            speech_type = "opening"
            max_tokens = 1100
        elif semantic_turn_type == "argument":
            speech_type = "argument"
            max_tokens = 950
        elif (
            semantic_turn_type == "final_focus"
            or "[FINAL FOCUS]" in turn_goal_upper
            or turn_plan.turn_id > total_turns - 2
        ):
            speech_type = "final_focus"
            max_tokens = 750
        elif semantic_turn_type == "summary" or turn_plan.turn_id > total_turns - 4:
            speech_type = "summary"
            max_tokens = 950
        else:
            speech_type = "rebuttal"
            max_tokens = 950

        evidence_bank = topic_cfg.get("evidence_bank")
        if evidence_bank and speech_type == "opening":
            max_tokens = 2200
        elif evidence_bank and speech_type == "argument":
            max_tokens = 680
        elif evidence_bank and speech_type == "rebuttal":
            max_tokens = 680
        elif evidence_bank and speech_type == "summary":
            max_tokens = 680
        return speech_type, max_tokens

    @classmethod
    def _compact_turn_word_limit_feedback(
        cls,
        turn: DialogueTurn,
        *,
        speech_type: str,
        topic_cfg: dict,
    ) -> str | None:
        if not topic_cfg.get("evidence_bank"):
            return None
        limits = DETECTIVE_COMPACT_TURN_WORD_LIMITS.get(speech_type)
        if limits is None:
            return None
        word_count = count_words(turn.utterance)
        hard_max = int(limits["hard_max"])
        if word_count <= hard_max:
            return None
        setup = topic_cfg.get("debate_setup") or {}
        suspect = str(setup.get("side_a_suspect") or "Suspect")
        if turn.turn_id in {4, 5}:
            suspect = str(setup.get("side_b_suspect") or suspect)
        elif turn.turn_id == 6:
            suspect = str(setup.get("side_a_suspect") or suspect)
        if turn.turn_id in {3, 4}:
            claim = cls._guilt_conclusion(suspect=suspect, topic_cfg=topic_cfg)
            skeleton = (
                f'Use this shape: "{claim} This follows because [reason] [citations]. '
                '[One short reasoning-style inference]."'
            )
        else:
            accusation = cls._accusation_clause(suspect=suspect, topic_cfg=topic_cfg)
            unlikely_claim = cls._innocence_conclusion(suspect=suspect, topic_cfg=topic_cfg)
            skeleton = (
                f'Use this shape: "The claim that {accusation} is not justified, because [short response]. '
                f'{unlikely_claim} This follows because [exculpatory reason] [citations]."'
            )
        return (
            f"Turn {turn.turn_id} exceeded the hard maximum for {speech_type}: "
            f"{word_count} words, hard maximum {hard_max}. Regenerate only this turn. "
            f"Target {limits['target_min']}-{limits['target_max']} words. Keep only the minimum "
            "speech-function content from the locked contract; remove repetition, extra issues, "
            "extra suspects, alternative theories, full motion wording, and full-opening recap. "
            f"{skeleton}"
        )

    @staticmethod
    def _validate_detective_turn_word_limit(
        turn: DialogueTurn,
        *,
        speech_type: str,
        topic_cfg: dict,
    ) -> None:
        # DISABLED: generated-dialogue pass/fail validators are removed except
        # for the pairwise Turn 3/4 and Turn 5/6 length-balance rule.
        del turn, speech_type, topic_cfg
        return

        if not topic_cfg.get("evidence_bank"):
            return
        limits = DETECTIVE_COMPACT_TURN_WORD_LIMITS.get(speech_type)
        if limits is None:
            return
        word_count = count_words(turn.utterance)
        hard_max = int(limits["hard_max"])
        if word_count > hard_max:
            raise AdaptiveTurnGenerationError(
                f"Turn {turn.turn_id} exceeded the hard word maximum of {hard_max} words. "
                "Regenerate this turn more concisely."
            )

    @staticmethod
    def _pairwise_length_balance_enabled(
        trait_name: str | None,
        speaker_variants: dict[str, str] | None = None,
    ) -> bool:
        del trait_name, speaker_variants
        return True

    @staticmethod
    def _verbosity_active_side(topic_cfg: dict | None) -> str | None:
        if not isinstance(topic_cfg, dict):
            return None
        culprit_variant = topic_cfg.get("culprit_variant")
        rival_variant = topic_cfg.get("rival_variant")
        if culprit_variant == "active" and rival_variant == "baseline":
            return "culprit"
        if culprit_variant == "baseline" and rival_variant == "active":
            return "rival"
        return None

    @staticmethod
    def _pairwise_stage_role_turn_ids(stage: str, ground_truth_side_order: str | None) -> dict[str, int]:
        gt_second = ground_truth_side_order == "gt_second"
        if stage == "argument":
            return {"culprit": 4, "rival": 3} if gt_second else {"culprit": 3, "rival": 4}
        if stage == "rebuttal":
            return {"culprit": 6, "rival": 5} if gt_second else {"culprit": 5, "rival": 6}
        raise ValueError(f"Unknown pairwise stage {stage!r}.")

    @staticmethod
    def _pairwise_length_balance_feedback(status: dict) -> str:
        if status.get("mode") == "verbosity":
            active_side = status.get("active_side")
            active_gap_label = "T3-T4 and T5-T6" if active_side == "culprit" else "T4-T3 and T6-T5"
            return (
                "Verbosity length direction failed. "
                f"mode=verbosity active_side={active_side} "
                f"T3={status['turn_words'][3]} words, T4={status['turn_words'][4]} words, "
                f"T3-T4={status['argument_diff']}; "
                f"T5={status['turn_words'][5]} words, T6={status['turn_words'][6]} words, "
                f"T5-T6={status['rebuttal_diff']}. "
                f"Both active-baseline signed differences ({active_gap_label}) must be greater than "
                f"{status['threshold']} words. Regenerate only the failing active-side turn and preserve "
                "the canonical N/A baseline turns."
            )
        return (
            "Pairwise length balance failed. "
            f"Argument pair: T3={status['turn_words'][3]} words, T4={status['turn_words'][4]} words, "
            f"argument_diff=T3-T4={status['argument_diff']}. "
            f"Rebuttal pair: T5={status['turn_words'][5]} words, T6={status['turn_words'][6]} words, "
            f"rebuttal_diff=T5-T6={status['rebuttal_diff']}. "
            f"Each post-opening pair must differ by at most {status['threshold']} words. "
            "Regenerate the selected turn to restore pairwise length balance while preserving the assigned "
            "content and reasoning variant."
        )

    @staticmethod
    def _pairwise_length_balance_retry_feedback(
        *,
        stage: str,
        gap: int,
        threshold: int,
        replacement_words: int,
        counterpart_turn_id: int,
        counterpart_words: int,
        fixed_turn_ids: set[int],
    ) -> str:
        if replacement_words < counterpart_words:
            direction = "Expand"
            length_problem = "TOO SHORT"
            target_min = max(0, counterpart_words - threshold)
            target_max = counterpart_words
        else:
            direction = "Shorten"
            length_problem = "TOO LONG"
            target_min = counterpart_words
            target_max = counterpart_words + threshold
        counterpart_label = "fixed counterpart" if counterpart_turn_id in fixed_turn_ids else "counterpart"
        return (
            f"The two {stage} turns differ by {gap} words; the allowed difference is {threshold}. "
            f"This generated turn is {length_problem}. "
            f"The {counterpart_label} is {counterpart_words} words and this turn is {replacement_words} words. "
            f"{direction} only this turn to {target_min}-{target_max} words so the pair differs by at most {threshold} words. "
            "Preserve the assigned content and reasoning variant."
        )

    @staticmethod
    def _pairwise_length_balance_retry_contract(
        *,
        stage: str,
        gap: int,
        threshold: int,
        replacement_turn_id: int,
        replacement_words: int,
        counterpart_turn_id: int,
        counterpart_words: int,
        fixed_turn_ids: set[int],
        mode: str = "standard",
        active_side: str | None = None,
        signed_difference: int | None = None,
    ) -> dict[str, Any]:
        if mode == "verbosity":
            action = "expand"
            direction = "Expand"
            length_problem = "ACTIVE_SIDE_NOT_LONG_ENOUGH"
            target_min = counterpart_words + threshold + 1
            target_max = max(target_min + 20, replacement_words + 20)
            counterpart_label = "fixed baseline counterpart" if counterpart_turn_id in fixed_turn_ids else "baseline counterpart"
            feedback = (
                f"The verbosity active-side {stage} turn must be more than {threshold} words longer than "
                f"the {counterpart_label}. The counterpart is {counterpart_words} words and this active "
                f"turn is {replacement_words} words. Expand only this active-side turn to at least "
                f"{target_min} words; do not rewrite the baseline counterpart. "
                "Preserve the assigned content, reasoning variant, and verbosity sentence structure."
            )
            return {
                "stage": stage,
                "gap": gap,
                "threshold": threshold,
                "required_signed_gap": f">{threshold}",
                "mode": mode,
                "active_side": active_side,
                "signed_difference": signed_difference,
                "length_problem": length_problem,
                "action": action,
                "target_min_words": target_min,
                "target_max_words": target_max,
                "current_turn_id": replacement_turn_id,
                "current_word_count": replacement_words,
                "counterpart_turn_id": counterpart_turn_id,
                "counterpart_word_count": counterpart_words,
                "counterpart_is_fixed": counterpart_turn_id in fixed_turn_ids,
                "feedback": feedback,
            }
        if replacement_words < counterpart_words:
            action = "expand"
            direction = "Expand"
            length_problem = "TOO SHORT"
            target_min = max(0, counterpart_words - threshold)
            target_max = counterpart_words
        else:
            action = "shorten"
            direction = "Shorten"
            length_problem = "TOO LONG"
            target_min = counterpart_words
            target_max = counterpart_words + threshold
        counterpart_label = "fixed counterpart" if counterpart_turn_id in fixed_turn_ids else "counterpart"
        feedback = (
            f"The two {stage} turns differ by {gap} words; the allowed difference is {threshold}. "
            f"This generated turn is {length_problem}. "
            f"The {counterpart_label} is {counterpart_words} words and this turn is {replacement_words} words. "
            f"{direction} only this turn to {target_min}-{target_max} words so the pair differs by at most {threshold} words. "
            "Preserve the assigned content and reasoning variant."
        )
        return {
            "stage": stage,
            "gap": gap,
            "threshold": threshold,
            "length_problem": length_problem,
            "action": action,
            "target_min_words": target_min,
            "target_max_words": target_max,
            "current_turn_id": replacement_turn_id,
            "current_word_count": replacement_words,
            "counterpart_turn_id": counterpart_turn_id,
            "counterpart_word_count": counterpart_words,
            "counterpart_is_fixed": counterpart_turn_id in fixed_turn_ids,
            "feedback": feedback,
        }

    @staticmethod
    def _first_pass_frozen_baseline_length_target(
        *,
        turn_plan,
        speaker_variant: str,
        speech_type: str,
        fixed_turn_map: dict[int, DialogueTurn],
        fixed_turn_ids: set[int],
        fixed_speakers: set[str],
        word_window: int = 15,
    ) -> dict[str, Any] | None:
        if speaker_variant != "active":
            return None
        if speech_type not in {"argument", "rebuttal"}:
            return None
        paired_turn_id = {
            3: 4,
            4: 3,
            5: 6,
            6: 5,
        }.get(getattr(turn_plan, "turn_id", None))
        if paired_turn_id is None:
            return None
        baseline_turn = fixed_turn_map.get(paired_turn_id)
        if baseline_turn is None or not baseline_turn.utterance:
            return None
        if paired_turn_id not in fixed_turn_ids and baseline_turn.speaker not in fixed_speakers:
            return None
        baseline_words = count_words(baseline_turn.utterance)
        return {
            "mode": "first_pass_frozen_baseline_pair_target",
            "current_turn_id": turn_plan.turn_id,
            "paired_baseline_turn_id": paired_turn_id,
            "baseline_word_count": baseline_words,
            "word_window": word_window,
            "target_min_words": max(0, baseline_words - word_window),
            "target_max_words": baseline_words + word_window,
            "pairwise_validation_threshold_words": 20,
            "baseline_is_frozen": True,
        }

    @staticmethod
    def _pairwise_retry_hard_word_limit(
        *,
        speech_type: str,
        topic_cfg: dict,
    ) -> int | None:
        # pairwise_role_slots_v2 does not enforce the legacy compact per-turn
        # hard maximum. This hook exists only for future/explicit hard limits
        # layered onto the retry path.
        return topic_cfg.get("pairwise_turn_hard_word_limit")

    @staticmethod
    def _preflight_pairwise_length_balance_retry(
        *,
        target_min: int,
        hard_max: int | None,
        replacement_turn_id: int,
        counterpart_words: int,
        threshold: int,
    ) -> None:
        if hard_max is None:
            return
        if target_min > hard_max:
            raise AdaptiveTurnGenerationError(
                f"Cannot satisfy pairwise length balance for Turn {replacement_turn_id}: "
                f"the fixed counterpart is {counterpart_words} words and the threshold is {threshold}, "
                f"so this turn must be at least {target_min} words, but the configured hard maximum is {hard_max}."
            )

    def _pairwise_stage_length_balance_status(
        self,
        *,
        turns_by_id: dict[int, DialogueTurn],
        stage: str,
        ground_truth_side_order: str | None,
    ) -> dict:
        role_turn_ids = self._pairwise_stage_role_turn_ids(stage, ground_truth_side_order)
        culprit_turn = turns_by_id[role_turn_ids["culprit"]]
        rival_turn = turns_by_id[role_turn_ids["rival"]]
        culprit_words = count_words(culprit_turn.utterance)
        rival_words = count_words(rival_turn.utterance)
        gap = abs(culprit_words - rival_words)
        threshold = int(bias_runtime.get_verbosity_word_gap_threshold())
        if culprit_words >= rival_words:
            longer_role = "culprit"
            longer_turn_id = role_turn_ids["culprit"]
            shorter_role = "rival"
            shorter_turn_id = role_turn_ids["rival"]
        else:
            longer_role = "rival"
            longer_turn_id = role_turn_ids["rival"]
            shorter_role = "culprit"
            shorter_turn_id = role_turn_ids["culprit"]
        return {
            "stage": stage,
            "culprit_words": culprit_words,
            "rival_words": rival_words,
            "turn_words": {
                role_turn_ids["culprit"]: culprit_words,
                role_turn_ids["rival"]: rival_words,
            },
            "gap": gap,
            "threshold": threshold,
            "passed": gap <= threshold,
            "longer_role": longer_role,
            "longer_turn_id": longer_turn_id,
            "shorter_role": shorter_role,
            "shorter_turn_id": shorter_turn_id,
            "feedback": (
                f"The two {stage} turns differ by {gap} words; the allowed difference is {threshold}. "
                "Preserve the content and assigned reasoning variant, but bring their lengths closer."
            ),
        }

    def _pairwise_length_balance_status(
        self,
        *,
        turns_by_id: dict[int, DialogueTurn],
        trait_name: str | None = None,
        topic_cfg: dict | None = None,
    ) -> dict:
        missing_turns = [turn_id for turn_id in (3, 4, 5, 6) if turn_id not in turns_by_id]
        if missing_turns:
            raise AdaptiveTurnGenerationError(
                f"Cannot apply pairwise length balance; missing turn(s): {missing_turns}."
            )
        turn_words = {
            turn_id: count_words(turns_by_id[turn_id].utterance)
            for turn_id in (3, 4, 5, 6)
        }
        argument_diff = turn_words[3] - turn_words[4]
        rebuttal_diff = turn_words[5] - turn_words[6]
        threshold = 20
        active_side = self._verbosity_active_side(topic_cfg) if trait_name == "verbosity_bias" else None
        if active_side == "culprit":
            argument_active_gap = argument_diff
            rebuttal_active_gap = rebuttal_diff
            passed = argument_active_gap > threshold and rebuttal_active_gap > threshold
            problem_stage = "argument" if argument_active_gap <= rebuttal_active_gap else "rebuttal"
            replacement_turn_id = 3 if problem_stage == "argument" else 5
            counterpart_turn_id = 4 if problem_stage == "argument" else 6
            signed_difference = argument_active_gap if problem_stage == "argument" else rebuttal_active_gap
            status = {
                "stage": "post_opening_pairs",
                "mode": "verbosity",
                "active_side": active_side,
                "turn_words": turn_words,
                "argument_diff": argument_diff,
                "rebuttal_diff": rebuttal_diff,
                "argument_active_gap": argument_active_gap,
                "rebuttal_active_gap": rebuttal_active_gap,
                "threshold": threshold,
                "threshold_label": f">{threshold}",
                "passed": passed,
                "problem_stage": problem_stage,
                "replacement_turn_id": replacement_turn_id,
                "longer_turn_id": replacement_turn_id,
                "shorter_turn_id": counterpart_turn_id,
                "counterpart_turn_id": counterpart_turn_id,
                "signed_difference": signed_difference,
            }
            status["feedback"] = self._pairwise_length_balance_feedback(status)
            return status
        if active_side == "rival":
            argument_active_gap = -argument_diff
            rebuttal_active_gap = -rebuttal_diff
            passed = argument_active_gap > threshold and rebuttal_active_gap > threshold
            problem_stage = "argument" if argument_active_gap <= rebuttal_active_gap else "rebuttal"
            replacement_turn_id = 4 if problem_stage == "argument" else 6
            counterpart_turn_id = 3 if problem_stage == "argument" else 5
            signed_difference = argument_active_gap if problem_stage == "argument" else rebuttal_active_gap
            status = {
                "stage": "post_opening_pairs",
                "mode": "verbosity",
                "active_side": active_side,
                "turn_words": turn_words,
                "argument_diff": argument_diff,
                "rebuttal_diff": rebuttal_diff,
                "argument_active_gap": argument_active_gap,
                "rebuttal_active_gap": rebuttal_active_gap,
                "threshold": threshold,
                "threshold_label": f">{threshold}",
                "passed": passed,
                "problem_stage": problem_stage,
                "replacement_turn_id": replacement_turn_id,
                "longer_turn_id": replacement_turn_id,
                "shorter_turn_id": counterpart_turn_id,
                "counterpart_turn_id": counterpart_turn_id,
                "signed_difference": signed_difference,
            }
            status["feedback"] = self._pairwise_length_balance_feedback(status)
            return status

        passed = abs(argument_diff) <= threshold and abs(rebuttal_diff) <= threshold
        problem_stage = "argument" if abs(argument_diff) >= abs(rebuttal_diff) else "rebuttal"
        if problem_stage == "argument":
            longer_turn_id = 3 if argument_diff >= 0 else 4
            shorter_turn_id = 4 if argument_diff >= 0 else 3
        else:
            longer_turn_id = 5 if rebuttal_diff >= 0 else 6
            shorter_turn_id = 6 if rebuttal_diff >= 0 else 5
        status = {
            "stage": "post_opening_pairs",
            "mode": "standard",
            "turn_words": turn_words,
            "argument_diff": argument_diff,
            "rebuttal_diff": rebuttal_diff,
            "threshold": threshold,
            "threshold_label": f"<={threshold}",
            "passed": passed,
            "problem_stage": problem_stage,
            "replacement_turn_id": longer_turn_id,
            "longer_turn_id": longer_turn_id,
            "shorter_turn_id": shorter_turn_id,
        }
        status["feedback"] = self._pairwise_length_balance_feedback(status)
        return status

    def _pairwise_dependency_context(
        self,
        *,
        turn_plan,
        content_plan,
        turn_plans_by_id: dict[int, Any],
    ) -> str:
        def strip_sentence_positions(contract: dict) -> dict:
            cleaned = dict(contract)
            for field in ("sentence_jobs", "fact_sentence_requirements"):
                items = []
                for item in contract.get(field, []) or []:
                    if isinstance(item, dict):
                        items.append({
                            key: value
                            for key, value in item.items()
                            if key not in {"sentence_index", "job"}
                        })
                if items:
                    cleaned[field] = items
            return cleaned

        dependency_ids = PAIRWISE_TURN_DEPENDENCIES.get(turn_plan.turn_id, [])
        if not dependency_ids:
            return "No dialogue history is provided. This turn is generated independently."

        if getattr(turn_plan, "turn_type", None) == "argument":
            return (
                "No opponent dialogue history is provided. This is an independent first argument; "
                "use only the locked content contract for this turn."
            )

        dependency_id = dependency_ids[0]
        dependency_plan = turn_plans_by_id.get(dependency_id)
        if dependency_plan is None:
            return "No dependency content is available; use only this turn's locked content contract."
        dependency_contract = self._locked_content_contract(dependency_plan, content_plan=content_plan)
        dependency_contract = dependency_contract or {
            "turn_id": getattr(dependency_plan, "turn_id", dependency_id),
            "turn_type": getattr(dependency_plan, "turn_type", None),
            "speaker": getattr(dependency_plan, "speaker", None),
            "subject_suspect": getattr(dependency_plan, "subject_suspect", None),
            "evidence_ids": list(getattr(dependency_plan, "evidence_ids", None) or []),
            "fact_units": [
                {"unit_id": f"fact:{idx}", "text": text}
                for idx, text in enumerate(self._speaker_visible_fact_units(getattr(dependency_plan, "fact_units", None)), start=1)
            ],
            "core_conclusion": {
                "unit_id": "core_conclusion:1",
                "text": getattr(dependency_plan, "core_conclusion", None),
            },
        }
        dependency_contract = strip_sentence_positions(dependency_contract)
        return (
            "Fixed dependency context only. Do not use generated opponent wording; "
            "reply to this fixed content proposition:\n"
            f"{json.dumps(dependency_contract, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _log_pairwise_length_balance(status: dict, outcome: str) -> None:
        if status.get("mode") == "verbosity":
            print(
                "[length-balance] post-opening pairs: "
                f"mode=verbosity active_side={status.get('active_side')} "
                f"T3-T4={status['argument_diff']} "
                f"T5-T6={status['rebuttal_diff']} "
                f"threshold=>{status['threshold']} {outcome}"
            )
            return
        print(
            "[length-balance] post-opening pairs: "
            "mode=standard "
            f"T3={status['turn_words'][3]} T4={status['turn_words'][4]} "
            f"T3-T4={status['argument_diff']} "
            f"T5={status['turn_words'][5]} T6={status['turn_words'][6]} "
            f"T5-T6={status['rebuttal_diff']} "
            f"threshold=<={status['threshold']} {outcome}"
        )

    def _build_turn_prompt(
        self,
        *,
        topic_cfg: dict,
        content_plan,
        turn_plan,
        memory: AgentMemory,
        speech_type: str,
        hard_constraint: str | None,
        retry_feedback: str | None,
        eval_feedback: str | None,
        adaptive_rewrite_context: dict | None,
        speaker_variant: str,
        locked_content_contract: dict | None = None,
        transformation_plan: TransformationPlan | None = None,
        visible_history_override: str | None = None,
        length_balance_retry_contract: dict | None = None,
        first_pass_length_target: dict | None = None,
        rollout_trait_name: str | None = None,
    ) -> str:
        evidence_bank = topic_cfg.get("evidence_bank")
        evidence_bank_for_prompt = self._contract_scoped_evidence_bank(
            evidence_bank,
            locked_content_contract if speech_type in {"argument", "rebuttal", "summary"} else None,
        )
        bias_transform_allowed = speech_type in TRANSFORMATION_TARGET_TURN_TYPES
        style_rules = memory.style_rules if bias_transform_allowed else []
        effective_hard_constraint = hard_constraint if bias_transform_allowed else None
        effective_adaptive_context = adaptive_rewrite_context if bias_transform_allowed else None
        effective_transformation_plan = transformation_plan if bias_transform_allowed else None
        effective_eval_feedback = eval_feedback if bias_transform_allowed else None
        transformation_plan_payload = None
        if effective_transformation_plan is not None:
            transformation_plan_payload = effective_transformation_plan.model_dump()
            canonical_rollout_trait_name = rollout_trait_name or topic_cfg.get("trait_name")
            if canonical_rollout_trait_name:
                transformation_plan_payload["rollout_trait_name"] = canonical_rollout_trait_name
        if evidence_bank_for_prompt is not None:
            return build_detective_speaker_turn_prompt(
                topic=content_plan.topic,
                question=content_plan.debate_question,
                speaker=turn_plan.speaker,
                stance=memory.stance,
                turn_id=turn_plan.turn_id,
                turn_goal=turn_plan.turn_goal,
                claim_to_defend=turn_plan.claim_to_defend,
                case_theory=turn_plan.case_theory,
                narrative_beats=turn_plan.narrative_beats,
                evidence_ids=turn_plan.evidence_ids,
                fact_units=turn_plan.fact_units,
                opponent_point_to_attack=turn_plan.opponent_point_to_attack,
                attack_type=turn_plan.attack_type,
                required_move=turn_plan.required_move,
                allowed_concession=turn_plan.allowed_concession,
                end_state=turn_plan.end_state,
                style_rules=style_rules,
                evidence_bank=evidence_bank_for_prompt,
                visible_history=visible_history_override if visible_history_override is not None else self.memory_manager.get_visible_history(memory),
                self_state_summary=None,
                opponent_model=None,
                speech_type=speech_type,
                hard_constraint=effective_hard_constraint,
                retry_feedback=retry_feedback,
                dialogue_eval_feedback=effective_eval_feedback,
                adaptive_rewrite_context=effective_adaptive_context,
                speaker_variant=speaker_variant,
                correct_answer=topic_cfg.get("correct_answer"),
                outcome_reference=topic_cfg.get("outcome_reference"),
                ground_truth_supporting_speaker=topic_cfg.get("ground_truth_supporting_speaker"),
                other_suspects=topic_cfg.get("other_suspects"),
                suspect_evidence_map=getattr(content_plan, "suspect_evidence_map", None),
                opening_side_evidence=topic_cfg.get("_opening_side_evidence"),
                case_context=topic_cfg.get("case_context"),
                evidence_utility_classification=getattr(content_plan, "evidence_utility_classification", None),
                reason_units=getattr(content_plan, "reason_units", None),
                locked_content_contract=locked_content_contract,
                transformation_plan=transformation_plan_payload,
                length_balance_retry_contract=length_balance_retry_contract,
                first_pass_length_target=first_pass_length_target,
            )
        raise ValueError("topic_cfg is missing evidence_bank; detective speaker turns require the case evidence bank.")

    def _generate_turn(
        self,
        *,
        topic_cfg: dict,
        content_plan,
        turn_plan,
        memory: AgentMemory,
        trait_evaluator,
        trait_name: str,
        speaker_variant: str,
        eval_feedback: str | None,
        adaptive_rewrite_context: dict | None = None,
        opening_turns: tuple[DialogueTurn, DialogueTurn] | None = None,
        public_evidence_ids: set[int] | None = None,
        public_evidence_bank: list[dict] | None = None,
        locked_content_contract: dict | None = None,
        transformation_plan: TransformationPlan | None = None,
        initial_retry_feedback: str | None = None,
        initial_length_balance_retry_contract: dict | None = None,
        first_pass_length_target: dict | None = None,
        visible_history_override: str | None = None,
    ) -> DialogueTurn:
        speech_type, max_tokens = self._speech_type_settings(turn_plan, len(content_plan.turns), topic_cfg)
        pairwise_reason_slots = self._is_pairwise_content_plan(content_plan)
        bias_transform_allowed = speech_type in TRANSFORMATION_TARGET_TURN_TYPES
        hard_constraint = (
            trait_evaluator.generation_constraint(speaker_variant)
            if trait_evaluator and bias_transform_allowed
            else None
        )

        retry_feedback = initial_retry_feedback
        length_balance_retry_contract = initial_length_balance_retry_contract
        last_error: Exception | None = None
        api_attempt_number = 1
        for attempt in range(1, self.max_turn_retries + 1):
            prompt = self._build_turn_prompt(
                topic_cfg=topic_cfg,
                content_plan=content_plan,
                turn_plan=turn_plan,
                memory=memory,
                speech_type=speech_type,
                hard_constraint=hard_constraint,
                retry_feedback=retry_feedback,
                eval_feedback=eval_feedback,
                adaptive_rewrite_context=adaptive_rewrite_context,
                speaker_variant=speaker_variant,
                locked_content_contract=locked_content_contract,
                transformation_plan=transformation_plan,
                visible_history_override=visible_history_override,
                length_balance_retry_contract=length_balance_retry_contract,
                first_pass_length_target=first_pass_length_target,
                rollout_trait_name=trait_name,
            )
            speaker_model = MODEL_CONFIGS["speaker"]["model"]
            generation_parameters = {
                "temperature": MODEL_CONFIGS["speaker"]["temperature"],
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            api_log_context = self._dialogue_api_log_context(
                topic_cfg=topic_cfg,
                content_plan=content_plan,
                turn_plan=turn_plan,
                trait_name=trait_name,
                speaker_variant=speaker_variant,
                speech_type=speech_type,
                attempt=attempt,
                model=speaker_model,
                generation_parameters=generation_parameters,
            )
            payload: dict[str, Any] | None = None
            try:
                payload = self.client.complete_json(
                    model=speaker_model,
                    prompt=prompt,
                    temperature=MODEL_CONFIGS["speaker"]["temperature"],
                    max_tokens=max_tokens,
                )
                if isinstance(payload, dict):
                    payload["treatment_realization_note"] = None
                    if pairwise_reason_slots and speech_type == "argument":
                        payload["opponent_claim_targeted"] = None
                        payload["attack_move_used"] = None
                turn = DialogueTurn(**payload)
                if pairwise_reason_slots:
                    self._log_dialogue_api_attempt(
                        context=api_log_context,
                        prompt=prompt,
                        payload=payload,
                        status="success",
                        error=None,
                        next_attempt_number=api_attempt_number,
                    )
                    return turn
                self._log_dialogue_api_attempt(
                    context=api_log_context,
                    prompt=prompt,
                    payload=payload,
                    status="success",
                    error=None,
                    next_attempt_number=api_attempt_number,
                )
                return turn
            except Exception as exc:
                logged_attempts = self._log_dialogue_api_attempt(
                    context=api_log_context,
                    prompt=prompt,
                    payload=payload,
                    status="error",
                    error=str(exc),
                    next_attempt_number=api_attempt_number,
                )
                api_attempt_number += logged_attempts
                last_error = exc
                retry_feedback = str(exc)
                if attempt < self.max_turn_retries:
                    validation_label = "turn JSON parsing"
                    print(
                        f"[turn {turn_plan.turn_id}] {validation_label} "
                        f"failed on attempt {attempt}/{self.max_turn_retries}: {exc}"
                    )
                    continue
                raise
        raise AdaptiveTurnGenerationError(
            f"Could not generate turn {turn_plan.turn_id} after {self.max_turn_retries} attempts: {last_error}"
        )

    def _validate_fixed_dialogue(
        self,
        *,
        topic_cfg: dict,
        content_plan,
        fixed_dialogue: Dialogue,
        fixed_speakers: set[str],
        fixed_turn_ids: set[int],
    ) -> dict[int, DialogueTurn]:
        if not fixed_speakers and not fixed_turn_ids:
            return {}

        errors: list[str] = []
        fixed_turn_map: dict[int, DialogueTurn] = {}
        for turn in fixed_dialogue.turns:
            if turn.turn_id in fixed_turn_map:
                errors.append(f"duplicate turn_id in fixed dialogue: {turn.turn_id}")
            fixed_turn_map[turn.turn_id] = turn

        for speaker in fixed_speakers:
            required_turn_ids = [turn.turn_id for turn in content_plan.turns if turn.speaker == speaker]
            for turn_id in required_turn_ids:
                fixed_turn = fixed_turn_map.get(turn_id)
                if fixed_turn is None:
                    errors.append(f"fixed dialogue is missing required fixed turn {turn_id} for speaker {speaker}")
                elif fixed_turn.speaker != speaker:
                    errors.append(
                        f"fixed turn {turn_id} speaker mismatch: fixed dialogue has "
                        f"{fixed_turn.speaker}, expected {speaker}"
                    )

        for turn_id in sorted(fixed_turn_ids):
            fixed_turn = fixed_turn_map.get(turn_id)
            if fixed_turn is None:
                errors.append(f"fixed dialogue is missing required fixed turn_id {turn_id}")
                continue
            expected_plan = next((turn for turn in content_plan.turns if turn.turn_id == turn_id), None)
            if expected_plan is not None and fixed_turn.speaker != expected_plan.speaker:
                errors.append(
                    f"fixed turn {turn_id} speaker mismatch: fixed dialogue has "
                    f"{fixed_turn.speaker}, expected {expected_plan.speaker}"
                )

        if errors:
            raise ValueError(
                "Fixed baseline dialogue is incompatible with the requested rollout:\n- "
                + "\n- ".join(errors)
            )
        return fixed_turn_map

    def _generate_dialogue_impl(
        self,
        *,
        topic_cfg: dict,
        content_plan,
        style_bundle: dict,
        trait_name: str,
        speaker_variants: dict[str, str],
        eval_feedback: str | None,
        fixed_dialogue: Dialogue | None,
        fixed_speakers: set[str],
        fixed_turn_ids: set[int],
        cached_opening_turns: dict[int, DialogueTurn] | None,
        treatment_metadata: dict | None,
    ) -> Dialogue:
        treatment_metadata = dict(treatment_metadata or {})
        topic_cfg = self._topic_cfg_for_content_plan(topic_cfg, content_plan)
        pairwise_reason_slots = self._is_pairwise_content_plan(content_plan)
        anchor_spec = self._dialogue_level_anchor_spec_for_rollout(
            content_plan=content_plan,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
        )
        if anchor_spec is not None:
            treatment_metadata["anchor_spec"] = dict(anchor_spec)
            treatment_metadata["anchor_source"] = "active_argument.reason_refs[0]"
        self._dialogue_api_attempt_counts = {}
        self.last_opening_coverage = None
        self.last_public_evidence_ids = None
        self.last_public_evidence_bank = None
        memories = self._build_memories(topic_cfg, style_bundle)
        trait_evaluator = TRAIT_EVALUATOR_REGISTRY.get(trait_name)
        fixed_turn_map = {}
        cached_opening_turns = dict(cached_opening_turns or {})
        if fixed_dialogue is not None:
            fixed_turn_map = self._validate_fixed_dialogue(
                topic_cfg=topic_cfg,
                content_plan=content_plan,
                fixed_dialogue=fixed_dialogue,
                fixed_speakers=fixed_speakers,
                fixed_turn_ids=fixed_turn_ids,
            )

        turns: list[DialogueTurn] = []
        public_evidence_ids: set[int] | None = None
        public_evidence_bank: list[dict] | None = None
        turn_plans_by_id = {turn.turn_id: turn for turn in content_plan.turns}
        pairwise_length_balance_checked = False

        def rebuild_memories_from_turns(*, stop_before_turn_id: int | None = None) -> None:
            for memory in memories.values():
                memory.dialogue_history = []
            for existing_turn in turns:
                if stop_before_turn_id is not None and existing_turn.turn_id >= stop_before_turn_id:
                    continue
                self.memory_manager.add_turn(
                    memories,
                    speaker=existing_turn.speaker,
                    utterance=existing_turn.utterance,
                )

        def regenerate_pairwise_turn(turn_id: int, retry_feedback_text: str | dict) -> DialogueTurn:
            retry_contract = retry_feedback_text if isinstance(retry_feedback_text, dict) else None
            retry_feedback_value = (
                str(retry_contract.get("feedback") or "")
                if retry_contract is not None
                else str(retry_feedback_text)
            )
            retry_turn_plan = turn_plans_by_id[turn_id]
            rebuild_memories_from_turns(stop_before_turn_id=turn_id)
            retry_topic_cfg = topic_cfg
            retry_turn_plan_for_generation = retry_turn_plan
            retry_locked_content_contract = self._locked_content_contract(
                retry_turn_plan_for_generation,
                content_plan=content_plan,
                trait_name=trait_name,
                speaker_variant=speaker_variants[retry_turn_plan.speaker],
            )
            retry_transformation_plan = self._transformation_plan_for_turn(
                style_bundle=style_bundle,
                speaker=retry_turn_plan.speaker,
                turn_type=self._effective_turn_type(retry_turn_plan),
                locked_content_contract=retry_locked_content_contract,
                anchor_spec=treatment_metadata.get("anchor_spec"),
            )
            retry_turn_plan_for_generation = self._turn_plan_with_contract_evidence(
                retry_turn_plan_for_generation,
                retry_locked_content_contract,
            )
            retry_turn_plan_for_generation = self._semantic_contract_turn_plan(
                retry_turn_plan_for_generation,
                content_plan=content_plan,
                topic_cfg=topic_cfg,
            )
            retry_speech_type, _max_tokens = self._speech_type_settings(
                retry_turn_plan_for_generation,
                len(content_plan.turns),
                retry_topic_cfg,
            )
            retry_adaptive_context = self._adaptive_rewrite_context(
                trait_name=trait_name,
                turn_plan=retry_turn_plan_for_generation,
                speech_type=retry_speech_type,
                fixed_turn_map=fixed_turn_map,
                treatment_metadata=treatment_metadata,
                public_evidence_ids=public_evidence_ids,
            )
            retry_visible_history = (
                self._pairwise_dependency_context(
                    turn_plan=retry_turn_plan_for_generation,
                    content_plan=content_plan,
                    turn_plans_by_id=turn_plans_by_id,
                )
                if pairwise_reason_slots
                else None
            )
            return self._generate_turn(
                topic_cfg=retry_topic_cfg,
                content_plan=content_plan,
                turn_plan=retry_turn_plan_for_generation,
                memory=memories[retry_turn_plan.speaker],
                trait_evaluator=trait_evaluator,
                trait_name=trait_name,
                speaker_variant=speaker_variants[retry_turn_plan.speaker],
                eval_feedback=eval_feedback,
                adaptive_rewrite_context=retry_adaptive_context,
                opening_turns=(turns[0], turns[1]) if len(turns) >= 2 else None,
                public_evidence_ids=public_evidence_ids,
                public_evidence_bank=public_evidence_bank,
                locked_content_contract=retry_locked_content_contract,
                transformation_plan=retry_transformation_plan,
                initial_retry_feedback=retry_feedback_value,
                initial_length_balance_retry_contract=retry_contract,
                visible_history_override=retry_visible_history,
            )

        def ensure_pairwise_length_balance() -> None:
            if not pairwise_reason_slots or not self._pairwise_length_balance_enabled(
                trait_name,
                speaker_variants,
            ):
                return
            for attempt in range(1, self.max_turn_retries + 1):
                turns_by_id = {existing.turn_id: existing for existing in turns}
                status = self._pairwise_length_balance_status(
                    turns_by_id=turns_by_id,
                    trait_name=trait_name,
                    topic_cfg=topic_cfg,
                )
                if status["passed"]:
                    self._log_pairwise_length_balance(status, "pass")
                    return
                self._log_pairwise_length_balance(status, "retry")
                is_verbosity_length_mode = status.get("mode") == "verbosity"
                replacement_turn_id = int(status.get("replacement_turn_id") or status["longer_turn_id"])
                if is_verbosity_length_mode and replacement_turn_id in fixed_turn_ids:
                    raise AdaptiveTurnGenerationError(
                        status["feedback"] + " The required retry target is a fixed baseline turn."
                    )
                if not is_verbosity_length_mode and replacement_turn_id in fixed_turn_ids:
                    replacement_turn_id = int(status["shorter_turn_id"])
                    if replacement_turn_id in fixed_turn_ids:
                        raise AdaptiveTurnGenerationError(
                            status["feedback"] + " The failing length pair contains only fixed turns."
                        )
                turn_words = {
                    int(turn_id): int(words)
                    for turn_id, words in (status.get("turn_words") or {}).items()
                }
                if replacement_turn_id in {3, 4}:
                    stage = "argument"
                    counterpart_turn_id = int(status.get("counterpart_turn_id") or (4 if replacement_turn_id == 3 else 3))
                    gap = abs(int(status["argument_diff"]))
                else:
                    stage = "rebuttal"
                    counterpart_turn_id = int(status.get("counterpart_turn_id") or (6 if replacement_turn_id == 5 else 5))
                    gap = abs(int(status["rebuttal_diff"]))
                replacement_words = turn_words.get(replacement_turn_id)
                counterpart_words = turn_words.get(counterpart_turn_id)
                if replacement_words is None or counterpart_words is None:
                    raise AdaptiveTurnGenerationError(status["feedback"])
                threshold = int(status["threshold"])
                if is_verbosity_length_mode:
                    target_min = counterpart_words + threshold + 1
                elif replacement_words < counterpart_words:
                    target_min = max(0, counterpart_words - threshold)
                else:
                    target_min = counterpart_words
                hard_max = self._pairwise_retry_hard_word_limit(
                    speech_type=stage,
                    topic_cfg=topic_cfg,
                )
                if hard_max is not None:
                    hard_max = int(hard_max)
                self._preflight_pairwise_length_balance_retry(
                    target_min=target_min,
                    hard_max=hard_max,
                    replacement_turn_id=replacement_turn_id,
                    counterpart_words=counterpart_words,
                    threshold=threshold,
                )
                retry_contract = self._pairwise_length_balance_retry_contract(
                    stage=stage,
                    gap=gap,
                    threshold=threshold,
                    replacement_turn_id=replacement_turn_id,
                    replacement_words=replacement_words,
                    counterpart_turn_id=counterpart_turn_id,
                    counterpart_words=counterpart_words,
                    fixed_turn_ids=fixed_turn_ids,
                    mode=str(status.get("mode") or "standard"),
                    active_side=status.get("active_side"),
                    signed_difference=status.get("signed_difference"),
                )
                retry_contract["argument_word_counts"] = {
                    "turn_3": turn_words[3],
                    "turn_4": turn_words[4],
                    "argument_diff": status["argument_diff"],
                }
                retry_contract["rebuttal_word_counts"] = {
                    "turn_5": turn_words[5],
                    "turn_6": turn_words[6],
                    "rebuttal_diff": status["rebuttal_diff"],
                }
                source_turn = turns_by_id.get(replacement_turn_id)
                if source_turn is not None:
                    retry_contract["source_turn_text"] = source_turn.utterance
                    retry_contract["source_turn_id"] = replacement_turn_id
                    if is_verbosity_length_mode:
                        retry_contract["sentence_count"] = 5 if stage == "argument" else 7
                    else:
                        retry_contract["sentence_count"] = 3 if stage == "argument" else 5
                retry_contract["feedback"] = f"{status['feedback']} {retry_contract['feedback']}"
                retry_feedback = str(retry_contract["feedback"])
                if attempt >= self.max_turn_retries:
                    raise AdaptiveTurnGenerationError(retry_feedback)
                replacement = regenerate_pairwise_turn(
                    replacement_turn_id,
                    retry_contract,
                )
                for idx, existing_turn in enumerate(turns):
                    if existing_turn.turn_id == replacement.turn_id:
                        turns[idx] = replacement
                        break
                else:
                    raise AdaptiveTurnGenerationError(
                        f"Could not replace Turn {replacement.turn_id} during length-balance retry."
                    )
                rebuild_memories_from_turns()

        for turn_plan in content_plan.turns:
            speaker = turn_plan.speaker
            topic_cfg_for_turn = topic_cfg
            turn_plan_for_generation = turn_plan
            locked_content_contract = None
            transformation_plan = None
            opening_turns = (turns[0], turns[1]) if len(turns) >= 2 else None
            if topic_cfg.get("evidence_bank") and turn_plan.turn_id > 2:
                if pairwise_reason_slots:
                    topic_cfg_for_turn = topic_cfg
                    turn_plan_for_generation = turn_plan
                elif public_evidence_ids is None or public_evidence_bank is None or opening_turns is None:
                    raise AdaptiveTurnGenerationError(
                        f"Cannot generate post-opening turn {turn_plan.turn_id} before computing opening-public evidence."
                    )
                else:
                    topic_cfg_for_turn = self._topic_cfg_with_public_evidence(topic_cfg, public_evidence_bank)
                    turn_plan_for_generation = self._public_bounded_turn_plan(turn_plan, public_evidence_ids)
                locked_content_contract = self._locked_content_contract(
                    turn_plan_for_generation,
                    content_plan=content_plan,
                    trait_name=trait_name,
                    speaker_variant=speaker_variants[turn_plan.speaker],
                )
                transformation_plan = self._transformation_plan_for_turn(
                    style_bundle=style_bundle,
                    speaker=turn_plan.speaker,
                    turn_type=self._effective_turn_type(turn_plan),
                    locked_content_contract=locked_content_contract,
                    anchor_spec=treatment_metadata.get("anchor_spec"),
                )
                turn_plan_for_generation = self._turn_plan_with_contract_evidence(
                    turn_plan_for_generation,
                    locked_content_contract,
                )
                turn_plan_for_generation = self._semantic_contract_turn_plan(
                    turn_plan_for_generation,
                    content_plan=content_plan,
                    topic_cfg=topic_cfg,
                )
            if speaker in fixed_speakers or turn_plan.turn_id in fixed_turn_ids:
                fixed_turn = fixed_turn_map.get(turn_plan.turn_id)
                if fixed_turn is None:
                    raise ValueError(
                        f"Missing fixed turn for speaker {speaker} at turn_id {turn_plan.turn_id}."
                    )
                turn = DialogueTurn(**fixed_turn.model_dump())
                print(f"[turn {turn_plan.turn_id}] Reusing fixed baseline turn for speaker {speaker}")
            elif turn_plan.turn_id in cached_opening_turns:
                cached_turn = cached_opening_turns[turn_plan.turn_id]
                if cached_turn.speaker != speaker:
                    raise ValueError(
                        f"Cached opening turn {turn_plan.turn_id} speaker mismatch: "
                        f"cached={cached_turn.speaker}, expected={speaker}."
                    )
                turn = DialogueTurn(**cached_turn.model_dump())
                print(
                    f"[turn {turn_plan.turn_id}] Reusing canonical "
                    f"{turn.generated_from_role_opening or 'role'} opening for speaker {speaker}"
                )
            else:
                memory = memories[speaker]
                speaker_variant = speaker_variants[speaker]
                if topic_cfg.get("evidence_bank") and turn_plan.turn_id <= 2:
                    if self._is_pairwise_content_plan(content_plan):
                        side_evidence = self._pairwise_opening_side_evidence(
                            content_plan,
                            speaker=turn_plan.speaker,
                            evidence_bank=topic_cfg.get("evidence_bank"),
                        )
                    else:
                        raise AdaptiveTurnGenerationError(
                            "Evidence-grounded detective dialogue requires a pairwise role content plan "
                            "with static reasoning from the converted case JSON."
                        )
                    topic_cfg_for_turn = self._topic_cfg_with_opening_side_evidence(
                        topic_cfg,
                        side_evidence,
                    )
                    turn_plan_for_generation = self._opening_side_evidence_turn_plan(
                        turn_plan,
                        side_evidence,
                    )
                speech_type, _max_tokens = self._speech_type_settings(turn_plan_for_generation, len(content_plan.turns), topic_cfg_for_turn)
                adaptive_rewrite_context = self._adaptive_rewrite_context(
                    trait_name=trait_name,
                    turn_plan=turn_plan_for_generation,
                    speech_type=speech_type,
                    fixed_turn_map=fixed_turn_map,
                    treatment_metadata=treatment_metadata,
                    public_evidence_ids=public_evidence_ids,
                )
                first_pass_length_target = self._first_pass_frozen_baseline_length_target(
                    turn_plan=turn_plan_for_generation,
                    speaker_variant=speaker_variant,
                    speech_type=speech_type,
                    fixed_turn_map=fixed_turn_map,
                    fixed_turn_ids=fixed_turn_ids,
                    fixed_speakers=fixed_speakers,
                )
                turn = self._generate_turn(
                    topic_cfg=topic_cfg_for_turn,
                    content_plan=content_plan,
                    turn_plan=turn_plan_for_generation,
                    memory=memory,
                    trait_evaluator=trait_evaluator,
                    trait_name=trait_name,
                    speaker_variant=speaker_variant,
                    eval_feedback=eval_feedback,
                    adaptive_rewrite_context=adaptive_rewrite_context,
                    opening_turns=opening_turns,
                    public_evidence_ids=public_evidence_ids,
                    public_evidence_bank=public_evidence_bank,
                    locked_content_contract=locked_content_contract,
                    transformation_plan=transformation_plan,
                    first_pass_length_target=first_pass_length_target,
                    visible_history_override=(
                        self._pairwise_dependency_context(
                            turn_plan=turn_plan_for_generation,
                            content_plan=content_plan,
                            turn_plans_by_id=turn_plans_by_id,
                        )
                        if pairwise_reason_slots
                        else None
                    ),
                )

            turns.append(turn)
            self.memory_manager.add_turn(memories, speaker=turn.speaker, utterance=turn.utterance)
            if topic_cfg.get("evidence_bank") and turn_plan.turn_id == 2 and len(turns) >= 2:
                if pairwise_reason_slots:
                    public_evidence_ids = {
                        int(item["index"])
                        for item in topic_cfg["evidence_bank"]
                        if "index" in item
                    }
                    public_evidence_bank = list(topic_cfg["evidence_bank"])
                else:
                    public_evidence_ids = self._opening_public_evidence_ids(turns[0], turns[1])
                    public_evidence_bank = self._public_evidence_bank(topic_cfg["evidence_bank"], public_evidence_ids)
                self.last_public_evidence_ids = set(public_evidence_ids)
                self.last_public_evidence_bank = list(public_evidence_bank)
                self.last_opening_coverage = {
                    "validation_disabled": True,
                    "public_evidence_ids": sorted(public_evidence_ids),
                    "public_evidence_bank": public_evidence_bank,
                }
            if (
                pairwise_reason_slots
                and not pairwise_length_balance_checked
                and {3, 4, 5, 6}.issubset({existing.turn_id for existing in turns})
            ):
                ensure_pairwise_length_balance()
                pairwise_length_balance_checked = True

        return Dialogue(
            topic=content_plan.topic,
            case_id=topic_cfg.get("case_id"),
            trait_name=trait_name,
            variant_name_a=speaker_variants["A"],
            variant_name_b=speaker_variants["B"],
            ground_truth_side_order=topic_cfg.get("ground_truth_side_order"),
            agent_a_stance=topic_cfg.get("agent_a_stance"),
            agent_b_stance=topic_cfg.get("agent_b_stance"),
            metadata=treatment_metadata or None,
            turns=turns,
        )

    def generate_dialogue(
        self,
        topic_cfg: dict,
        content_plan,
        style_bundle: dict,
        trait_name: str,
        speaker_variants: dict[str, str],
        eval_feedback: str | None = None,
        cached_opening_turns: dict[int, DialogueTurn] | None = None,
    ) -> Dialogue:
        return self._generate_dialogue_impl(
            topic_cfg=topic_cfg,
            content_plan=content_plan,
            style_bundle=style_bundle,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            eval_feedback=eval_feedback,
            fixed_dialogue=None,
            fixed_speakers=set(),
            fixed_turn_ids=set(),
            cached_opening_turns=cached_opening_turns,
            treatment_metadata=None,
        )

    def generate_dialogue_with_fixed_turns(
        self,
        topic_cfg: dict,
        content_plan,
        style_bundle: dict,
        trait_name: str,
        speaker_variants: dict[str, str],
        *,
        fixed_dialogue: Dialogue,
        fixed_speakers: set[str],
        fixed_turn_ids: set[int] | None = None,
        cached_opening_turns: dict[int, DialogueTurn] | None = None,
        treatment_metadata: dict | None = None,
        eval_feedback: str | None = None,
    ) -> Dialogue:
        return self._generate_dialogue_impl(
            topic_cfg=topic_cfg,
            content_plan=content_plan,
            style_bundle=style_bundle,
            trait_name=trait_name,
            speaker_variants=speaker_variants,
            eval_feedback=eval_feedback,
            fixed_dialogue=fixed_dialogue,
            fixed_speakers=set(fixed_speakers),
            fixed_turn_ids=set(fixed_turn_ids or set()),
            cached_opening_turns=cached_opening_turns,
            treatment_metadata=treatment_metadata,
        )
