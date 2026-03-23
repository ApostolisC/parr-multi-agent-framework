"""
Weaviate RAG tools for the research assistant example.

Connects to a local Weaviate instance to provide keyword (BM25) search
and document retrieval.  No external dependencies beyond stdlib — uses
urllib + JSON directly against the Weaviate REST / GraphQL API.

Usage:
    from examples.research_assistant.rag_tools import (
        build_rag_tools, SEARCH_KB_TOOL, GET_DOCUMENT_TOOL,
    )

Environment:
    WEAVIATE_URL  — Weaviate base URL (default: http://localhost:8080)
    WEAVIATE_COLLECTION — Collection/class name (default: Dpia_documents)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from parr import Phase, ToolDef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "Dpia_documents")

# Properties to fetch from Weaviate (based on schema)
_CONTENT_PROPS = [
    "content",
    "filename",
    "file_type",
    "page_number",
    "chunk_index",
    "project_id",
    "artifact_id",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _graphql_query(query: str) -> Dict[str, Any]:
    """Execute a GraphQL query against Weaviate (sync)."""
    url = f"{WEAVIATE_URL}/v1/graphql"
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.error("Weaviate GraphQL request failed: %s", e)
        return {"errors": [{"message": str(e)}]}


def _rest_get(path: str) -> Dict[str, Any]:
    """GET from Weaviate REST API (sync)."""
    url = f"{WEAVIATE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.error("Weaviate REST request failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    project_id: Optional[str] = None,
) -> str:
    """
    BM25 keyword search across the knowledge base.

    Returns ranked chunks with content previews and metadata.
    """
    props = " ".join(_CONTENT_PROPS)

    # Build optional where filter for project_id
    where_clause = ""
    if project_id:
        where_clause = (
            f', where: {{path: ["project_id"], '
            f'operator: Equal, valueInt: {project_id}}}'
        )

    gql = (
        "{"
        f"Get{{{WEAVIATE_COLLECTION}("
        f'bm25: {{query: "{_escape_gql(query)}"}}, '
        f"limit: {top_k}"
        f"{where_clause}"
        f") {{{props} _additional {{id score}}}}}}"
        "}"
    )

    result = await asyncio.to_thread(_graphql_query, gql)

    # Handle errors
    if "errors" in result:
        error_msg = "; ".join(e.get("message", "?") for e in result["errors"])
        return json.dumps({"error": error_msg, "results": []})

    raw_items = (
        result.get("data", {})
        .get("Get", {})
        .get(WEAVIATE_COLLECTION, [])
    )

    results = []
    for item in raw_items:
        additional = item.get("_additional", {})
        content = item.get("content", "")
        # Build a summary (first 300 chars)
        summary = content[:300] + ("..." if len(content) > 300 else "")

        results.append({
            "document_id": additional.get("id", ""),
            "filename": item.get("filename", ""),
            "file_type": item.get("file_type", ""),
            "page_number": item.get("page_number"),
            "chunk_index": item.get("chunk_index"),
            "summary": summary,
            "relevance_score": additional.get("score"),
        })

    return json.dumps({
        "query": query,
        "results": results,
        "total": len(results),
    })


async def get_document(document_id: str) -> str:
    """
    Retrieve the full content of a document chunk by its Weaviate ID.
    """
    path = f"/v1/objects/{WEAVIATE_COLLECTION}/{document_id}"
    result = await asyncio.to_thread(_rest_get, path)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    props = result.get("properties", {})
    return json.dumps({
        "document_id": document_id,
        "content": props.get("content", ""),
        "filename": props.get("filename", ""),
        "file_type": props.get("file_type", ""),
        "page_number": props.get("page_number"),
        "chunk_index": props.get("chunk_index"),
        "project_id": props.get("project_id", ""),
        "artifact_id": props.get("artifact_id", ""),
    })


def _escape_gql(text: str) -> str:
    """Escape a string for use inside GraphQL double-quoted string."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


# ---------------------------------------------------------------------------
# ToolDef objects
# ---------------------------------------------------------------------------

SEARCH_KB_TOOL = ToolDef(
    name="search_knowledge_base",
    description=(
        "Search the knowledge base for relevant document chunks. "
        "Uses BM25 keyword matching. Returns ranked results with "
        "content previews, filenames, and relevance scores. "
        "Use this to find information about a topic."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
    handler=search_knowledge_base,
    phase_availability=[Phase.ACT],
    is_read_only=True,
)

GET_DOCUMENT_TOOL = ToolDef(
    name="get_document",
    description=(
        "Retrieve the full content of a document chunk by its ID "
        "(returned by search_knowledge_base). Use this to read the "
        "complete text of a search result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The document ID from search_knowledge_base results.",
            },
        },
        "required": ["document_id"],
    },
    handler=get_document,
    phase_availability=[Phase.ACT],
    is_read_only=True,
)


def build_rag_tools() -> Dict[str, ToolDef]:
    """Return a dict of RAG tools ready for the tool_registry."""
    return {
        "search_knowledge_base": SEARCH_KB_TOOL,
        "get_document": GET_DOCUMENT_TOOL,
    }
