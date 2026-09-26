"""
Base protocol for connectors (Telegram, Discord, etc.)
"""

from typing import Protocol, Callable, Awaitable
from datetime import datetime
from pydantic import BaseModel


class IncomingMessage(BaseModel):
    """Message received from external channel"""
    text: str
    user_id: str
    channel_id: str
    timestamp: datetime


# Callback type for message handlers
MessageCallback = Callable[[IncomingMessage], Awaitable[None]]


class Connector(Protocol):
    """Protocol defining the interface for communication connectors"""

    async def start(self) -> None:
        """
        Start the connector and begin listening for messages
        """
        ...

    async def stop(self) -> None:
        """
        Stop the connector gracefully
        """
        ...

    async def send(self, channel_id: str, text: str) -> None:
        """
        Send a message to a specific channel/chat

        Args:
            channel_id: Target channel/chat identifier
            text: Message text to send
        """
        ...

    async def send_photo(self, channel_id: str, image: bytes) -> None:
        """
        Send an image to a specific channel/chat, with no caption - the
        photo engine sends frames silently; the prompt behind one is
        inspected via the /photos command instead.

        Args:
            channel_id: Target channel/chat identifier
            image: Raw image bytes (e.g. PNG)
        """
        ...

    async def send_chat_action(self, channel_id: str, action: str) -> None:
        """
        Signal a transient status (e.g. "uploading a photo") while a slow
        reply is being prepared. Purely cosmetic - callers should not let a
        failure here interrupt the actual send.

        Args:
            channel_id: Target channel/chat identifier
            action: Action to signal, e.g. "upload_photo"
        """
        ...

    def on_message(self, callback: MessageCallback) -> None:
        """
        Register a callback to handle incoming messages

        Args:
            callback: Async function to call when message arrives
        """
        ...
