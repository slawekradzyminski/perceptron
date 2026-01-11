import pytest
from fastapi.testclient import TestClient

from backend.api_app import app


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with fresh dependency state.

    This fixture clears any dependency overrides before and after each test,
    ensuring tests don't interfere with each other.
    """
    app.dependency_overrides = {}
    yield TestClient(app)
    app.dependency_overrides = {}
