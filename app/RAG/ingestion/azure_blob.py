from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from ...config.settings import settings
from ...config.service_bootstrap import (
    create_blob_service_client,
    get_blob_container_name,
)
from pathlib import Path


class AzureBlobService:
    def __init__(self, create_container_if_missing: bool = True):
        self.container_name = get_blob_container_name()
        self.blob_service_client = create_blob_service_client()
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

        if create_container_if_missing:
            self.ensure_container_exists()

    def ensure_container_exists(self):
        try:
            self.container_client.create_container()
        except ResourceExistsError:
            pass

    def list_blobs(self) -> list[str]:
        return [blob.name for blob in self.container_client.list_blobs()]

    def download_blob(self, blob_name: str) -> bytes:
        blob_client = self.container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        return downloader.readall()

    def upload_blob(self, file, blob_name: str | None = None):
        blob_name = blob_name or Path(file).name
        blob_client = self.container_client.get_blob_client(blob_name)

        with open(file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        return {
            "blob_name": blob_name,
            "container_name": self.container_name,
        }

    def delete_blob(self, blob_name: str):
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            blob_client.delete_blob()

            return {
                "deleted": True,
                "blob_name": blob_name,
                "container_name": self.container_name,
            }

        except ResourceNotFoundError:
            return {
                "deleted": False,
                "message": "Blob not found",
                "blob_name": blob_name,
                "container_name": self.container_name,
            }


if __name__ == "__main__":
    blob_service = AzureBlobService(create_container_if_missing=True)
    file_path = settings.DATA_FILES[0]

    result = blob_service.upload_blob(file_path)
    print(f"Uploaded {file_path} to container {result['container_name']}")
    print(f"Blob name: {result['blob_name']}")
