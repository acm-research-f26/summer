"""
Hook to inject ARP hints into browser-use agent observations.

Usage (from web-ui repo root, after installing implementation_2 deps):

    from implementation_2.integrate_arp import enable_arp_layer
    enable_arp_layer(top_k=5)

This monkey-patches the agent's state message builder so each step prepends
ranked element hints. Does not replace the LLM — only augments context.
"""

from __future__ import annotations

import logging
from typing import Any

from implementation_2.src.action_relevance import ActionRelevanceScorer
from implementation_2.src.observation_enhancer import ObservationEnhancer

logger = logging.getLogger(__name__)

_enhancer: ObservationEnhancer | None = None
_original_get_state_message = None


def _parse_elements_from_state(state_message: str) -> list[dict]:
    """
    Extract interactive elements from browser-use state text.

    browser-use formats elements like: [index]<tag>text</tag>
    """
    import re

    elements = []
    pattern = re.compile(r"\[(\d+)\]<(\w+)>(.*?)</\2>", re.DOTALL)
    for match in pattern.finditer(state_message):
        elements.append({
            "index": int(match.group(1)),
            "tag": match.group(2),
            "text": match.group(3).strip(),
            "attrs": {},
        })
    return elements


def enable_arp_layer(top_k: int = 5, min_score: float = 0.05) -> None:
    """Enable ARP context injection on browser-use Agent."""
    global _enhancer, _original_get_state_message

    if _enhancer is not None:
        logger.warning("ARP layer already enabled")
        return

    try:
        from browser_use.agent.message_manager.service import MessageManager
    except ImportError as e:
        raise ImportError(
            "browser-use not installed. Run from web-ui venv with requirements.txt."
        ) from e

    scorer = ActionRelevanceScorer(top_k=top_k, min_score=min_score)
    _enhancer = ObservationEnhancer(scorer)

    _original_get_state_message = MessageManager.create_state_messages

    def patched_create_state_messages(self, *args: Any, **kwargs: Any):
        messages = _original_get_state_message(self, *args, **kwargs)
        task = getattr(self, "task", None) or ""
        if not task or not messages:
            return messages

        # Augment the last user/state message
        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)

        elements = _parse_elements_from_state(text)
        if elements:
            enhanced, _ranked = _enhancer.enhance(task, text, elements)
            if hasattr(last, "content"):
                last.content = enhanced
        return messages

    MessageManager.create_state_messages = patched_create_state_messages
    logger.info("ARP layer enabled (top_k=%d)", top_k)


def disable_arp_layer() -> None:
    """Restore original browser-use behavior."""
    global _enhancer, _original_get_state_message

    if _original_get_state_message is None:
        return

    from browser_use.agent.message_manager.service import MessageManager

    MessageManager.create_state_messages = _original_get_state_message
    _enhancer = None
    _original_get_state_message = None
    logger.info("ARP layer disabled")
