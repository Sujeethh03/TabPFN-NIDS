"""Event bus for real-time WebSocket notifications and live event broadcasting."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous in-memory event bus with subscriber queue management.

    Supports:
    - Broadcasting typed messages to all connected WebSocket clients.
    - Preserving a bounded history of recent detections for initial client synchronization.
    - Graceful connection add/remove.
    """

    def __init__(self, history_limit: int = 50) -> None:
        self.history_limit = history_limit
        self._recent_detections: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=history_limit
        )
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new subscriber and return its event queue."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        logger.debug("New subscriber connected. Total subscribers: %d", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Unregister a subscriber queue."""
        self._subscribers.discard(q)
        logger.debug("Subscriber disconnected. Total subscribers: %d", len(self._subscribers))

    def get_recent_history(self) -> list[dict[str, Any]]:
        """Return a copy of recent detection events."""
        return list(self._recent_detections)

    async def broadcast_detection(self, detection_data: dict[str, Any]) -> None:
        """Record and broadcast a detection event."""
        self._recent_detections.append(detection_data)
        await self.broadcast("detection", detection_data)

    async def broadcast_status(self, status_data: dict[str, Any]) -> None:
        """Broadcast a live capture status update."""
        await self.broadcast("capture_status", status_data)

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send an event payload to all connected subscribers."""
        if not self._subscribers:
            return

        payload = {"type": event_type, "data": data}
        dead_subscribers: list[asyncio.Queue[dict[str, Any]]] = []

        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # If a client is too slow, drop the oldest event to make room
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    dead_subscribers.append(q)
            except Exception:
                dead_subscribers.append(q)

        for dead in dead_subscribers:
            self._subscribers.discard(dead)
