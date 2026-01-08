"""Tests for /transformer API routes (Chapter 5)."""

from fastapi.testclient import TestClient

from backend.api_app import app

client = TestClient(app)


class TestTransformerState:
    def test_get_state(self) -> None:
        response = client.get("/transformer/state")
        assert response.status_code == 200
        data = response.json()
        assert "current_text" in data
        assert "vocab_size" in data
        assert "encoding_name" in data


class TestTransformerTokenize:
    def test_tokenize_simple(self) -> None:
        response = client.post("/transformer/tokenize", json={"text": "Hello world!"})
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world!"
        assert "token_count" in data
        assert "tokens" in data

    def test_tokenize_token_structure(self) -> None:
        response = client.post("/transformer/tokenize", json={"text": "Test"})
        assert response.status_code == 200
        data = response.json()
        for token in data["tokens"]:
            assert "id" in token
            assert "text" in token
            assert "position" in token

    def test_tokenize_missing_text(self) -> None:
        response = client.post("/transformer/tokenize", json={})
        assert response.status_code == 400
        assert "Missing 'text'" in response.json()["detail"]

    def test_tokenize_invalid_type(self) -> None:
        response = client.post("/transformer/tokenize", json={"text": 123})
        assert response.status_code == 400
        assert "'text' must be a string" in response.json()["detail"]

    def test_tokenize_too_long(self) -> None:
        long_text = "a" * 10001
        response = client.post("/transformer/tokenize", json={"text": long_text})
        assert response.status_code == 400
        assert "too long" in response.json()["detail"]


class TestTransformerEmbed:
    def test_embed_with_text(self) -> None:
        response = client.post("/transformer/embed", json={"text": "Hello world!"})
        assert response.status_code == 200
        data = response.json()
        assert "matrix_shape" in data
        assert "d_model" in data
        assert "explanation" in data

    def test_embed_without_text(self) -> None:
        # First tokenize something
        client.post("/transformer/tokenize", json={"text": "Previous text"})
        # Then embed without text
        response = client.post("/transformer/embed", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Previous text"


class TestTransformerTrace:
    def test_trace_with_text(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Hello world!", "n_blocks": 6})
        assert response.status_code == 200
        data = response.json()
        assert data["n_blocks"] == 6
        assert len(data["blocks"]) == 6
        assert "explanation" in data

    def test_trace_default_blocks(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Test"})
        assert response.status_code == 200
        data = response.json()
        assert data["n_blocks"] == 12  # Default

    def test_trace_invalid_n_blocks(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Test", "n_blocks": 0})
        assert response.status_code == 400

    def test_trace_too_many_blocks(self) -> None:
        response = client.post("/transformer/trace", json={"text": "Test", "n_blocks": 100})
        assert response.status_code == 400


class TestTransformerChat:
    def test_chat_simple(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        response = client.post("/transformer/chat", json={"messages": messages})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_missing_messages(self) -> None:
        response = client.post("/transformer/chat", json={})
        assert response.status_code == 400
        assert "Missing 'messages'" in response.json()["detail"]

    def test_chat_invalid_message_format(self) -> None:
        response = client.post("/transformer/chat", json={"messages": [{"wrong": "format"}]})
        assert response.status_code == 400


class TestTransformerOllamaStatus:
    def test_ollama_status(self) -> None:
        response = client.get("/transformer/ollama-status")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "model" in data


class TestTransformerModels:
    def test_get_models(self) -> None:
        response = client.get("/transformer/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 3
        assert "explanation" in data


class TestScaleEndpoints:
    def test_scale_all_models(self) -> None:
        response = client.get("/transformer/scale")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_count" in data

    def test_scale_cnns(self) -> None:
        response = client.get("/transformer/scale/cnns")
        assert response.status_code == 200
        data = response.json()
        for model in data["models"]:
            assert model["type"] == "CNN"

    def test_scale_transformers(self) -> None:
        response = client.get("/transformer/scale/transformers")
        assert response.status_code == 200
        data = response.json()
        for model in data["models"]:
            assert model["type"] == "Transformer"

    def test_scale_compare(self) -> None:
        response = client.get("/transformer/scale/compare?model1=AlexNet&model2=GPT-4")
        assert response.status_code == 200
        data = response.json()
        assert "ratio" in data
        assert "explanation" in data

    def test_scale_compare_invalid(self) -> None:
        response = client.get("/transformer/scale/compare?model1=Invalid&model2=GPT-4")
        assert response.status_code == 400

    def test_scale_growth(self) -> None:
        response = client.get("/transformer/scale/growth")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "insight" in data

