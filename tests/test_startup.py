import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI

from app.config import app_startup
from app.config.app_startup import (
    ensure_ingestion_resources,
    initialize_application_services,
    lifespan,
)


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_application_services_bootstraps_resources(self) -> None:
        fake_chat_service = Mock()
        fake_chat_service.history_store = SimpleNamespace(
            ensure_db_and_container=AsyncMock()
        )

        with (
            patch.object(
                type(app_startup.settings), "require", autospec=True
            ) as require,
            patch(
                "app.config.app_startup.asyncio.to_thread",
                new=AsyncMock(return_value=None),
            ) as to_thread,
            patch(
                "app.config.app_startup.MultiAgentCVChat",
                return_value=fake_chat_service,
            ),
        ):
            result = await initialize_application_services()

        self.assertIs(result, fake_chat_service)
        require.assert_called_once_with(
            app_startup.settings,
            *app_startup.REQUIRED_CAPABILITIES,
        )
        self.assertIs(to_thread.await_args.args[0], ensure_ingestion_resources)
        fake_chat_service.history_store.ensure_db_and_container.assert_awaited_once_with()

    async def test_lifespan_sets_chat_service_on_app_state(self) -> None:
        app = FastAPI()
        fake_chat_service = Mock(close=AsyncMock())

        with patch(
            "app.config.app_startup.initialize_application_services",
            new=AsyncMock(return_value=fake_chat_service),
        ):
            async with lifespan(app):
                self.assertIs(app.state.chat_service, fake_chat_service)

        fake_chat_service.close.assert_awaited_once_with()
