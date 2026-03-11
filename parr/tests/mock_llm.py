"""
Mock LLM for framework testing.

Provides a script-driven ToolCallingLLM implementation that returns
pre-configured responses in order. Supports both flat sequences and
phase-keyed response maps.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

from parr.core_types import (
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Response factories
# ---------------------------------------------------------------------------

def make_text_response(
    content: str,
    input_tokens: int = 50,
    output_tokens: int = 100,
) -> LLMResponse:
    """Create a text-only LLM response (no tool calls)."""
    return LLMResponse(
        content=content,
        tool_calls=None,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        raw_message=Message(role=MessageRole.ASSISTANT, content=content),
    )


def make_tool_call_response(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None,
    call_id: Optional[str] = None,
    input_tokens: int = 50,
    output_tokens: int = 80,
) -> LLMResponse:
    """Create an LLM response with a single tool call."""
    tc = ToolCall(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=arguments or {},
    )
    return LLMResponse(
        content=content,
        tool_calls=[tc],
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        raw_message=Message(
            role=MessageRole.ASSISTANT, content=content, tool_calls=[tc]
        ),
    )


def make_multi_tool_response(
    calls: List[Dict[str, Any]],
    content: Optional[str] = None,
    input_tokens: int = 50,
    output_tokens: int = 100,
) -> LLMResponse:
    """
    Create an LLM response with multiple tool calls.

    Args:
        calls: List of dicts with 'name' and optionally 'arguments', 'call_id'.
    """
    tool_calls = [
        ToolCall(
            id=c.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
            name=c["name"],
            arguments=c.get("arguments", {}),
        )
        for c in calls
    ]
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        raw_message=Message(
            role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls
        ),
    )


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockToolCallingLLM:
    """
    Script-driven mock LLM for testing.

    Accepts a flat list of responses OR a phase-keyed dict. Responses
    are returned in order. Tracks all calls for post-test assertions.

    Usage (flat list):
        llm = MockToolCallingLLM([
            make_text_response("Plan complete."),
            make_text_response("Act complete."),
            make_text_response("REVIEW_PASSED"),
            make_text_response("Report submitted."),
        ])

    Usage (phase-keyed):
        from parr import Phase
        llm = MockToolCallingLLM({
            Phase.PLAN: [make_text_response("Plan done.")],
            Phase.ACT: [
                make_tool_call_response("log_finding", {"category": "risk", ...}),
                make_text_response("Act done."),
            ],
            Phase.REVIEW: [make_text_response("REVIEW_PASSED")],
            Phase.REPORT: [make_text_response("Final report.")],
        })
    """

    def __init__(
        self,
        responses: Union[List[LLMResponse], Dict[Any, List[LLMResponse]]],
    ) -> None:
        if isinstance(responses, dict):
            self._phase_responses: Optional[Dict[Any, List[LLMResponse]]] = responses
            self._flat_responses: Optional[List[LLMResponse]] = None
            # Track index per phase
            self._phase_index: Dict[Any, int] = {k: 0 for k in responses}
        else:
            self._flat_responses = list(responses)
            self._phase_responses = None
            self._flat_index = 0

        self.call_count: int = 0
        self.calls_log: List[Dict[str, Any]] = []

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token: Any = None,
    ) -> LLMResponse:
        """Return the next scripted response."""
        self.call_count += 1
        self.calls_log.append({
            "call_number": self.call_count,
            "message_count": len(messages),
            "tool_count": len(tools),
            "model": model,
            "tool_names": [t.get("name", "?") for t in tools],
        })

        if self._phase_responses is not None:
            return self._get_phase_response(messages)
        else:
            return self._get_flat_response()

    def _get_flat_response(self) -> LLMResponse:
        """Get next response from flat list."""
        if self._flat_index >= len(self._flat_responses):
            # Default: return text to stop the loop
            return make_text_response(
                f"[MockLLM: exhausted {len(self._flat_responses)} scripted responses]"
            )
        response = self._flat_responses[self._flat_index]
        self._flat_index += 1
        return response

    def _get_phase_response(self, messages: List[Message]) -> LLMResponse:
        """Get next response based on detected phase from system prompt."""
        phase = self._detect_phase(messages)
        if phase is None or phase not in self._phase_responses:
            return make_text_response("[MockLLM: unknown phase]")

        idx = self._phase_index.get(phase, 0)
        phase_list = self._phase_responses[phase]

        if idx >= len(phase_list):
            return make_text_response(
                f"[MockLLM: exhausted {len(phase_list)} responses for {phase}]"
            )
        response = phase_list[idx]
        self._phase_index[phase] = idx + 1
        return response

    @staticmethod
    def _detect_phase(messages: List[Message]) -> Optional[Any]:
        """Detect the current phase from the system prompt content."""
        from parr.core_types import Phase

        for msg in messages:
            if msg.role == MessageRole.SYSTEM and msg.content:
                content_lower = msg.content.lower()
                if "phase: planning" in content_lower or "create_todo_list" in content_lower:
                    return Phase.PLAN
                if "phase: execution" in content_lower or "log_finding" in content_lower:
                    return Phase.ACT
                if "phase: review" in content_lower or "review_checklist" in content_lower:
                    return Phase.REVIEW
                if "phase: reporting" in content_lower or "submit_report" in content_lower:
                    return Phase.REPORT
        return None
