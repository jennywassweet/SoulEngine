"""
SoulEngine main entry point
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

from engine.llm.base import LLMProvider
from engine.llm.mistral import MistralProvider
from engine.llm.openrouter import OpenRouterProvider
from engine.connectors.telegram import TelegramConnector
from engine.config_utils import resolve_llm_config, runtime_dir
from engine.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

LLM_PROVIDERS: dict[str, type] = {
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
}


async def load_config() -> dict:
    """Load soul configuration from YAML"""
    config_path = Path(__file__).parent.parent / "soul" / "config.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


async def main():
    """Main engine loop"""
    logger.info("=" * 60)
    logger.info("SoulEngine is alive")
    logger.info("=" * 60)

    # Load configuration
    config = await load_config()
    base_path = Path(__file__).parent.parent
    setting = config.get("setting", "default")
    llm_config = resolve_llm_config(base_path, setting, config.get("llm", {}))
    logger.info(f"Active setting: {setting} — llm config: {llm_config}")

    # Initialize LLM provider
    provider_name = llm_config.get("provider", "mistral")
    provider_cls = LLM_PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown llm.provider '{provider_name}' — must be one of {list(LLM_PROVIDERS)}"
        )
    logger.info(f"Initializing LLM provider: {provider_name}")
    provider_kwargs = {}
    if llm_config.get("model"):
        provider_kwargs["model"] = llm_config["model"]
    llm: LLMProvider = provider_cls(**provider_kwargs)

    # Initialize Telegram connector
    logger.info("Initializing Telegram connector...")
    connector = TelegramConnector()

    # Initialize orchestrator
    logger.info("Initializing orchestrator...")

    # Paths — everything lives under the active setting
    runtime = runtime_dir(base_path, setting)
    chat_log_path = runtime / "chat_log.jsonl"
    state_path = runtime / "state.md"
    user_model_path = runtime / "user_model.md"

    orchestrator = Orchestrator(
        llm=llm,
        connector=connector,
        base_path=base_path,
        config=config,  # Pass full config for compression
        chat_log_path=chat_log_path,
        state_path=state_path,
        user_model_path=user_model_path,
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 2000),
        reasoning_effort=llm_config.get("reasoning_effort"),
        provider_routing=llm_config.get("provider_routing"),
    )

    # Start the bot
    logger.info("=" * 60)
    logger.info("Starting Telegram bot...")
    logger.info("Send a message to start chatting!")
    logger.info("=" * 60)

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await orchestrator.stop()
        logger.info("SoulEngine shutdown")


if __name__ == "__main__":
    asyncio.run(main())
