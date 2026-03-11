"""
DocumentSearchProvider Adapter for the Agentic Framework.

Wraps an existing RAG service to implement the framework's
DocumentSearchProvider protocol. Maps between RAG-specific types
(RetrievedChunk, RAGContext) and the framework's generic dict format.

Usage:
    adapter = RAGDocumentSearchAdapter(
        rag_service=my_rag_service,
        project_id="proj_123",
    )

    # Wire into orchestrator as search_documents / read_document_section
    orchestrator = Orchestrator(llm=llm, ...)
    # The framework builds tools from this adapter automatically
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum number of cached sections before eviction
_CACHE_MAX_SIZE = 1000


def _run_sync_in_async(coro_or_func, *args):
    """Run a synchronous function from an async context using run_in_executor.

    Avoids the deprecated ``asyncio.get_event_loop()`` by using
    ``asyncio.get_running_loop()`` when available.
    """
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, coro_or_func, *args)


class RAGDocumentSearchAdapter:
    """
    DocumentSearchProvider adapter that wraps a RAG service.

    Translates between the framework's ``search()`` / ``get_section()``
    protocol and the application's RAG service (e.g., Weaviate-backed
    retrieval). Handles sync→async bridging via ``run_in_executor``.
    """

    def __init__(
        self,
        rag_service: Any,
        project_id: str,
        retrieval_method: str = "hybrid",
        tenant_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            rag_service: A RAG service instance with ``retrieve_context()`` method.
            project_id: Project ID for tenant-scoped document retrieval.
            retrieval_method: Search strategy — "hybrid" or "vector".
            tenant_config: Optional tenant configuration passed to the RAG service.
        """
        self._rag_service = rag_service
        self._project_id = project_id
        self._retrieval_method = retrieval_method
        self._tenant_config = tenant_config or {"allow_external_llm": True}
        # Simple cache: section_id → chunk data (from recent searches)
        self._chunk_cache: Dict[str, Dict[str, Any]] = {}

    # -- DocumentSearchProvider protocol -------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents by semantic query.

        Delegates to the RAG service's ``retrieve_context()`` and maps
        the results to the framework's expected dict format.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return.
            filters: Structured filters (passed as metadata to RAG).

        Returns:
            List of dicts with: section_id, source_file, section_title,
            summary, relevance_score, metadata.
        """
        try:
            context = await _run_sync_in_async(
                lambda: self._rag_service.retrieve_context(
                    query=query,
                    project_id=self._project_id,
                    top_k=top_k,
                    retrieval_method=self._retrieval_method,
                    tenant_config=self._tenant_config,
                ),
            )
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for chunk in context.chunks:
            section_id = f"{chunk.artifact_id}:{chunk.chunk_index}"

            entry = {
                "section_id": section_id,
                "source_file": chunk.filename,
                "section_title": f"Chunk {chunk.chunk_index} from {chunk.filename}",
                "summary": chunk.content[:500] if len(chunk.content) > 500 else chunk.content,
                "relevance_score": chunk.score,
                "metadata": {
                    "file_type": chunk.file_type,
                    "page_number": chunk.page_number,
                    "artifact_id": chunk.artifact_id,
                    "chunk_index": chunk.chunk_index,
                },
            }
            results.append(entry)

            # Cache full chunk data for get_section() (with eviction)
            if len(self._chunk_cache) >= _CACHE_MAX_SIZE:
                self._chunk_cache.clear()
            self._chunk_cache[section_id] = {
                "section_id": section_id,
                "source_file": chunk.filename,
                "section_title": f"Chunk {chunk.chunk_index} from {chunk.filename}",
                "full_text": chunk.content,
                "metadata": {
                    "file_type": chunk.file_type,
                    "page_number": chunk.page_number,
                    "artifact_id": chunk.artifact_id,
                    "chunk_index": chunk.chunk_index,
                    "char_offset_start": chunk.char_offset_start,
                    "char_offset_end": chunk.char_offset_end,
                    "score": chunk.score,
                },
            }

        logger.debug(
            f"RAG search returned {len(results)} results for "
            f"project {self._project_id}: '{query[:50]}...'"
        )
        return results

    async def get_section(self, section_id: str) -> Dict[str, Any]:
        """
        Get the full text of a specific document section.

        First checks the in-memory cache (populated by ``search()``).
        If not cached, attempts to re-fetch from the RAG service.

        Args:
            section_id: Section identifier in "artifact_id:chunk_index" format.

        Returns:
            Dict with: section_id, source_file, section_title, full_text, metadata.
        """
        # Check cache first
        if section_id in self._chunk_cache:
            return self._chunk_cache[section_id]

        # Parse section_id and attempt re-fetch
        parts = section_id.split(":", 1)
        if len(parts) != 2:
            return {
                "section_id": section_id,
                "source_file": "unknown",
                "section_title": "Unknown section",
                "full_text": f"Section '{section_id}' not found. Use search_documents first.",
                "metadata": {},
            }

        artifact_id, chunk_index_str = parts
        try:
            chunk_index = int(chunk_index_str)
        except ValueError:
            return {
                "section_id": section_id,
                "source_file": "unknown",
                "section_title": "Invalid section ID",
                "full_text": f"Invalid section_id format: '{section_id}'",
                "metadata": {},
            }

        # Try to re-fetch by searching with a broad query scoped to this artifact
        try:
            context = await _run_sync_in_async(
                lambda: self._rag_service.retrieve_context(
                    query=f"content from artifact {artifact_id}",
                    project_id=self._project_id,
                    top_k=20,
                    retrieval_method="vector",
                    tenant_config=self._tenant_config,
                ),
            )

            for chunk in context.chunks:
                if chunk.artifact_id == artifact_id and chunk.chunk_index == chunk_index:
                    result = {
                        "section_id": section_id,
                        "source_file": chunk.filename,
                        "section_title": f"Chunk {chunk.chunk_index} from {chunk.filename}",
                        "full_text": chunk.content,
                        "metadata": {
                            "file_type": chunk.file_type,
                            "page_number": chunk.page_number,
                            "artifact_id": chunk.artifact_id,
                            "chunk_index": chunk.chunk_index,
                        },
                    }
                    self._chunk_cache[section_id] = result
                    return result

        except Exception as e:
            logger.warning(f"Re-fetch for section '{section_id}' failed: {e}")

        return {
            "section_id": section_id,
            "source_file": "unknown",
            "section_title": "Section not found",
            "full_text": (
                f"Could not retrieve section '{section_id}'. "
                f"It may have been evicted from cache. Use search_documents to find it again."
            ),
            "metadata": {},
        }

    # -- cache management ----------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the section cache."""
        self._chunk_cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached sections."""
        return len(self._chunk_cache)
