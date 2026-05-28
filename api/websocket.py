"""WebSocket server for real-time dashboard push updates."""

import asyncio
import json
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.connections: set = set()
        self.update_interval: int = 60
        self._running = False

    async def register(self, websocket) -> None:
        self.connections.add(websocket)
        logger.info(f"WebSocket connected: {id(websocket)} ({len(self.connections)} total)")

    async def unregister(self, websocket) -> None:
        self.connections.discard(websocket)
        logger.info(f"WebSocket disconnected: {id(websocket)} ({len(self.connections)} remaining)")

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast a message to all connected WebSocket clients."""
        disconnected = set()
        payload = json.dumps(message, default=str)

        for ws in self.connections:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            await self.unregister(ws)

        return len(self.connections)

    @property
    def connected_count(self) -> int:
        return len(self.connections)

    async def start_auto_broadcast(
        self,
        status_callback: Callable[[], Dict[str, Any]],
        alert_callback: Callable[[], Dict[str, Any]],
    ):
        """Continuously broadcast status and alerts at set intervals."""
        self._running = True
        while self._running:
            if self.connections:
                try:
                    status = status_callback()
                    await self.broadcast({
                        "type": "status_update",
                        "data": status,
                        "timestamp": datetime.now().isoformat(),
                    })
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")

            await asyncio.sleep(self.update_interval)

    def stop(self):
        self._running = False
        logger.info("WebSocket auto-broadcast stopped")
