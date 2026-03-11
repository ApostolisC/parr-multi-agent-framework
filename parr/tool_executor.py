"""
Tool Executor for the Agentic Framework.

Dispatches tool calls to their registered handlers, enforces per-tool
timeouts, tracks call counts for rate limiting, validates inputs against
JSON Schema before execution, validates outputs against output_schema
after execution, supports configurable retry, and wraps document-sourced
results in <untrusted_document_content> tags automatically.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate

from .core_types import Phase, ToolCall, ToolDef, ToolResult
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Legacy hardcoded set of tool names whose results contain user-uploaded
# document content.  New tools should set wraps_untrusted_content=True on
# their ToolDef instead.  This set is kept for backward compatibility.
DOCUMENT_CONTENT_TOOLS = frozenset({
    "search_documents",
    "read_document_section",
})


class ToolExecutor:
    """
    Executes tool calls against registered handlers.

    Responsibilities:
    - Dispatch tool calls to the correct handler
    - Validate input arguments against the tool's parameters JSON Schema
    - Enforce per-tool timeouts
    - Retry on transient failures (when configured)
    - Validate handler output against the tool's output_schema JSON Schema
    - Track call counts for max_calls_per_phase enforcement
    - Wrap document content in <untrusted_document_content> tags
    - Return structured ToolResult (never raises on tool failure)
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        # Track calls per (phase, tool_name) for rate limiting
        self._call_counts: Dict[str, int] = {}
        self._current_phase: Optional[Phase] = None

    def set_phase(self, phase: Phase) -> None:
        """Set the current phase and reset call counts."""
        self._current_phase = phase
        self._call_counts.clear()

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: The tool invocation from the LLM.

        Returns:
            ToolResult with success/failure and content.
        """
        tool_def = self._registry.get(tool_call.name)

        if tool_def is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Unknown tool: '{tool_call.name}'. "
                       f"Available tools: {self._registry.tool_names}",
            )

        # Phase availability check
        if self._current_phase and self._current_phase not in tool_def.phase_availability:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' is not available in the "
                       f"'{self._current_phase.value}' phase. "
                       f"Available phases: {[p.value for p in tool_def.phase_availability]}",
            )

        # Rate limit check
        if tool_def.max_calls_per_phase is not None:
            count = self._call_counts.get(tool_call.name, 0)
            if count >= tool_def.max_calls_per_phase:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    content="",
                    error=f"Tool '{tool_call.name}' has reached its maximum of "
                           f"{tool_def.max_calls_per_phase} calls in the "
                           f"'{self._current_phase.value}' phase.",
                )

        # Orchestrator tools should not be executed here
        if tool_def.is_orchestrator_tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' requires orchestrator handling. "
                       f"This is a framework bug — orchestrator should have intercepted this.",
            )

        # Check handler exists
        if tool_def.handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Tool '{tool_call.name}' has no handler registered.",
            )

        # --- INPUT VALIDATION ---
        input_error = self._validate_input(tool_def, tool_call.arguments)
        if input_error:
            logger.warning(input_error)
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=input_error,
            )

        # --- EXECUTE WITH RETRY ---
        max_attempts = 1
        if tool_def.retry_on_failure and tool_def.max_retries > 0:
            max_attempts = 1 + tool_def.max_retries

        last_error: Optional[str] = None
        for attempt in range(max_attempts):
            try:
                result_content = await self._call_handler(
                    tool_def, tool_call.arguments
                )

                # --- OUTPUT VALIDATION ---
                output_error = self._validate_output(tool_def, result_content)
                if output_error:
                    logger.warning(output_error)
                    # Output validation failure = handler bug, NOT retried
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        content="",
                        error=output_error,
                    )

                # Track call count
                self._call_counts[tool_call.name] = (
                    self._call_counts.get(tool_call.name, 0) + 1
                )

                # Serialize output
                serialized = self._serialize_output(tool_def, result_content)

                # Wrap document content (check ToolDef flag + legacy set)
                if tool_def.wraps_untrusted_content or tool_call.name in DOCUMENT_CONTENT_TOOLS:
                    serialized = self._wrap_untrusted(serialized)

                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=True,
                    content=serialized,
                )

            except asyncio.TimeoutError:
                last_error = (
                    f"Tool '{tool_call.name}' timed out after "
                    f"{tool_def.timeout_ms}ms."
                )
                logger.warning(last_error)
                if attempt < max_attempts - 1:
                    backoff_delay = 2 ** attempt
                    logger.info(
                        f"Retrying tool '{tool_call.name}' "
                        f"(attempt {attempt + 2}/{max_attempts}) "
                        f"after {backoff_delay}s backoff"
                    )
                    await asyncio.sleep(backoff_delay)
                    continue

            except Exception as e:
                last_error = f"Tool '{tool_call.name}' failed: {str(e)}"
                logger.error(last_error, exc_info=True)
                if attempt < max_attempts - 1:
                    backoff_delay = 2 ** attempt
                    logger.info(
                        f"Retrying tool '{tool_call.name}' "
                        f"(attempt {attempt + 2}/{max_attempts}) "
                        f"after {backoff_delay}s backoff"
                    )
                    await asyncio.sleep(backoff_delay)
                    continue

        # All attempts exhausted
        return ToolResult(
            tool_call_id=tool_call.id,
            success=False,
            content="",
            error=last_error,
        )

    # -------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _validate_input(tool_def: ToolDef, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Validate tool arguments against the tool's parameters JSON Schema.

        Returns None if valid, or an error message string if invalid.
        """
        params = tool_def.parameters
        if not params or not params.get("properties"):
            return None
        try:
            jsonschema_validate(instance=arguments, schema=params)
            return None
        except JsonSchemaValidationError as e:
            return (
                f"Input validation failed for tool '{tool_def.name}': {e.message}"
            )

    @staticmethod
    def _validate_output(tool_def: ToolDef, result: Any) -> Optional[str]:
        """
        Validate tool handler output against the tool's output_schema JSON Schema.

        Returns None if valid, or an error message string if invalid.
        Only called when tool_def.output_schema is not None.
        """
        if tool_def.output_schema is None:
            return None
        try:
            jsonschema_validate(instance=result, schema=tool_def.output_schema)
            return None
        except JsonSchemaValidationError as e:
            return (
                f"Output validation failed for tool '{tool_def.name}': {e.message}"
            )

    @staticmethod
    def _serialize_output(tool_def: ToolDef, result: Any) -> str:
        """
        Serialize handler output to string for the LLM.

        When output_schema is set and result is a dict/list, uses json.dumps
        for proper JSON formatting. Otherwise falls back to str().
        """
        if tool_def.output_schema is not None and isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)

    # -------------------------------------------------------------------
    # Handler dispatch
    # -------------------------------------------------------------------

    async def _call_handler(
        self, tool_def: ToolDef, arguments: Dict[str, Any]
    ) -> Any:
        """Call the tool handler with timeout."""
        timeout_s = tool_def.timeout_ms / 1000.0

        handler = tool_def.handler
        if inspect.iscoroutinefunction(handler):
            result = await asyncio.wait_for(
                handler(**arguments), timeout=timeout_s
            )
        else:
            # Sync handlers run in executor to avoid blocking
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: handler(**arguments)),
                timeout=timeout_s,
            )

        return result

    @staticmethod
    def _wrap_untrusted(content: Any) -> str:
        """Wrap content in untrusted document content tags."""
        return (
            "<untrusted_document_content>\n"
            f"{content}\n"
            "</untrusted_document_content>"
        )
