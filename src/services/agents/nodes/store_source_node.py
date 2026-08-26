import logging
from typing import Dict, List

from langchain_core.messages import ToolMessage

from ..models import SourceItem
from ..state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_store_retrieved_sources_step(
    state: AgentState,
) -> Dict[str, List[SourceItem]]:
    """Extract source metadata from the retrieve_papers tool result."""

    logger.info("NODE: store_retrieved_sources")

    sources: List[SourceItem] = []

    # Find the latest retrieve_papers ToolMessage
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name == "retrieve_papers":
            content = message.content

            logger.debug(f"Tool result type: {type(content)}")

            # ToolNode may serialize the returned Documents.
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        metadata = item.get("metadata", {})

                        sources.append(
                            SourceItem(
                                arxiv_id=metadata.get("arxiv_id", ""),
                                title=metadata.get("title", ""),
                                authors=(
                                    metadata.get("authors", [])
                                    if isinstance(metadata.get("authors", []), list)
                                    else []
                                ),
                                url=metadata.get("source", ""),
                                relevance_score=float(
                                    metadata.get("score", 0.0) or 0.0
                                ),
                            )
                        )

            break

    logger.info(f"Stored {len(sources)} retrieved sources")

    return {
        "relevant_sources": sources,
    }