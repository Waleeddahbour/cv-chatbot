from uuid import uuid4
from datetime import datetime, timezone
import logging

from azure.cosmos import exceptions, PartitionKey

from ..config.settings import settings
from ..config.service_bootstrap import create_cosmos_client

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CosmosChatHistory:
    def __init__(self):
        self.client = create_cosmos_client()

        self.database_name = settings.AZURE_COSMOS_DATABASE_NAME
        self.container_name = settings.AZURE_COSMOS_CONTAINER_NAME

        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)

    async def close(self) -> None:
        await self.client.close()

    async def ensure_db_and_container(self, partition_key: str = "/session_id") -> None:
        """Create the Cosmos DB database and container if they do not exist."""
        try:
            self.database = await self.client.create_database_if_not_exists(
                id=self.database_name
            )

            self.container = await self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path=partition_key),
            )

            logger.info(
                "Cosmos DB ready. database=%s container=%s",
                self.database_name,
                self.container_name,
            )

        except exceptions.CosmosHttpResponseError as e:
            logger.exception("Error creating Cosmos database/container: %s", e.message)
            raise

    async def get_session(self, session_id: str) -> dict | None:
        """Get a chat session by session_id."""
        if not session_id:
            raise ValueError("session_id cannot be empty")

        try:
            return await self.container.read_item(
                item=session_id,
                partition_key=session_id,
            )

        except exceptions.CosmosResourceNotFoundError:
            return None

        except exceptions.CosmosHttpResponseError as e:
            logger.exception(
                "Error getting Cosmos session %s: %s", session_id, e.message
            )
            raise

    async def get_messages(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict]:
        """Get OpenAI-ready chat messages for a session."""
        session = await self.get_session(session_id)
        if not session:
            return []

        messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in session.get("history", [])
            if message.get("role") in {"user", "assistant"} and message.get("content")
        ]

        if max_messages is not None and max_messages > 0:
            return messages[-max_messages:]

        return messages

    async def create_session(self, session_id: str | None = None) -> dict:
        """Create a new empty chat session."""
        now = utc_now()

        if not session_id:
            session_id = f"session-{uuid4()}"

        session = {
            "id": session_id,
            "session_id": session_id,
            "history": [],
            "created_at": now,
            "updated_at": now,
        }

        try:
            return await self.container.create_item(session)

        except exceptions.CosmosHttpResponseError as e:
            logger.exception(
                "Error creating Cosmos session %s: %s", session_id, e.message
            )
            raise

    async def append_message(
        self,
        session_id: str | None,
        content: str,
        role: str,
    ) -> dict:
        """Append a message to a session. If the session does not exist, create it."""
        if not content:
            raise ValueError("content cannot be empty")

        if role not in {"user", "assistant"}:
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'.")

        now = utc_now()

        if not session_id:
            session = await self.create_session()
        else:
            session = await self.get_session(session_id)

            if not session:
                session = await self.create_session(session_id)

        session["history"].append({"role": role, "content": content, "created_at": now})
        session["updated_at"] = now

        try:
            return await self.container.upsert_item(session)

        except exceptions.CosmosHttpResponseError as e:
            logger.exception(
                "Error appending message to Cosmos session %s: %s",
                session_id,
                e.message,
            )
            raise

    async def delete_session(self, session_id: str) -> None:
        """Delete a session by session_id."""
        if not session_id:
            raise ValueError("session_id cannot be empty")

        try:
            await self.container.delete_item(
                item=session_id,
                partition_key=session_id,
            )

        except exceptions.CosmosResourceNotFoundError:
            pass

        except exceptions.CosmosHttpResponseError as e:
            logger.exception(
                "Error deleting Cosmos session %s: %s", session_id, e.message
            )
            raise


async def main() -> None:
    chat_history = CosmosChatHistory()
    try:
        await chat_history.ensure_db_and_container()
    finally:
        await chat_history.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
