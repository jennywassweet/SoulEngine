"""
Telegram connector implementation using aiogram 3.x
"""

import os
import logging
from datetime import datetime
from typing import Set

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode

from engine.connectors.base import Connector, IncomingMessage, MessageCallback

logger = logging.getLogger(__name__)


class TelegramConnector:
    """Telegram bot connector using aiogram"""

    def __init__(
        self,
        token: str | None = None,
        allowed_users: Set[int] | None = None,
    ):
        """
        Initialize Telegram connector

        Args:
            token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            allowed_users: Set of allowed user IDs (defaults to TELEGRAM_ALLOWED_USERS env var)
        """
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN must be provided or set in environment")

        # Parse allowed users from env if not provided
        if allowed_users is None:
            allowed_users_str = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
            if allowed_users_str:
                try:
                    allowed_users = {int(uid.strip()) for uid in allowed_users_str.split(",")}
                except ValueError as e:
                    raise ValueError(
                        f"TELEGRAM_ALLOWED_USERS must contain numeric user IDs, not usernames.\n"
                        f"Got: {allowed_users_str}\n"
                        f"To get your numeric ID, message @userinfobot in Telegram.\n"
                        f"Example: TELEGRAM_ALLOWED_USERS=123456789"
                    ) from e
            else:
                allowed_users = set()

        self.allowed_users = allowed_users
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()
        self._message_callback: MessageCallback | None = None

        # Dedup: track recently processed message IDs to avoid duplicates
        self._processed_message_ids: set[int] = set()
        self._max_processed_ids = 1000  # Keep last N to prevent memory leak

        # Register handlers
        self._setup_handlers()

        logger.info(
            f"Initialized TelegramConnector with {len(self.allowed_users)} allowed users"
        )

    def _setup_handlers(self):
        """Setup message and command handlers"""

        @self.dp.message(Command("start"))
        async def cmd_start(message: types.Message):
            """Handle /start command"""
            if not self._is_user_allowed(message.from_user.id):
                await message.answer("⛔️ Access denied. You are not authorized to use this bot.")
                logger.warning(f"Unauthorized access attempt from user {message.from_user.id}")
                return

            await message.answer(
                "👋 Hello! I'm alive and ready to chat.\n\n"
                "Just send me a message and I'll respond!"
            )

        @self.dp.message()
        async def handle_message(message: types.Message):
            """Handle all incoming messages"""
            if not self._is_user_allowed(message.from_user.id):
                await message.answer("⛔️ Access denied.")
                return

            # Dedup: skip if we already processed this message
            if message.message_id in self._processed_message_ids:
                logger.warning(f"Duplicate message_id {message.message_id}, skipping")
                return
            self._processed_message_ids.add(message.message_id)
            # Prevent memory leak by trimming old IDs
            if len(self._processed_message_ids) > self._max_processed_ids:
                # Remove oldest entries (set doesn't preserve order, so just clear half)
                to_keep = sorted(self._processed_message_ids)[-self._max_processed_ids // 2 :]
                self._processed_message_ids = set(to_keep)

            # Only process text messages
            if not message.text:
                await message.answer("I can only process text messages for now.")
                return

            # Call the registered callback
            if self._message_callback:
                incoming = IncomingMessage(
                    text=message.text,
                    user_id=str(message.from_user.id),
                    channel_id=str(message.chat.id),
                    timestamp=datetime.now(),
                )
                await self._message_callback(incoming)

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot"""
        # If no allowed users configured, allow everyone
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    def on_message(self, callback: MessageCallback) -> None:
        """Register callback for incoming messages"""
        self._message_callback = callback
        logger.info("Message callback registered")

    async def send(self, channel_id: str, text: str) -> None:
        """
        Send message to Telegram chat

        Args:
            channel_id: Telegram chat ID
            text: Message text (supports Markdown)
        """
        try:
            await self.bot.send_message(
                chat_id=int(channel_id),
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.debug(f"Sent message to {channel_id}")
        except Exception as e:
            logger.error(f"Error sending message to {channel_id}: {e}")
            # Fallback to plain text if Markdown fails
            try:
                await self.bot.send_message(
                    chat_id=int(channel_id),
                    text=text,
                )
            except Exception as e2:
                logger.error(f"Error sending plain text message: {e2}")
                raise

    async def send_photo(self, channel_id: str, image: bytes) -> None:
        """
        Send an image to a Telegram chat, with no caption

        Args:
            channel_id: Telegram chat ID
            image: Raw image bytes (e.g. PNG)
        """
        try:
            await self.bot.send_photo(
                chat_id=int(channel_id),
                photo=types.BufferedInputFile(image, filename="photo.png"),
            )
            logger.debug(f"Sent photo to {channel_id}")
        except Exception as e:
            logger.error(f"Error sending photo to {channel_id}: {e}")
            raise

    async def send_chat_action(self, channel_id: str, action: str) -> None:
        """
        Signal a transient status (e.g. "uploading a photo") in a Telegram chat

        Args:
            channel_id: Telegram chat ID
            action: Action to signal, e.g. "upload_photo" - passed straight
                through to Telegram's chat action API
        """
        try:
            await self.bot.send_chat_action(chat_id=int(channel_id), action=action)
        except Exception as e:
            logger.warning(f"Error sending chat action to {channel_id}: {e}")

    async def start(self) -> None:
        """Start the bot"""
        logger.info("Starting Telegram bot...")
        await self.dp.start_polling(self.bot)

    async def stop(self) -> None:
        """Stop the bot gracefully"""
        logger.info("Stopping Telegram bot...")
        await self.bot.session.close()
