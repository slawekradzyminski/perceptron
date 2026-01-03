import pytest
from fastapi.testclient import TestClient

from backend.api_app import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
