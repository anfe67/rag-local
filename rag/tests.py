import json
import shutil
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(
    RAG_DATA_DIR="/tmp/test_rag_data",
    OLLAMA_BASE_URL="http://127.0.0.1:11434",
    OLLAMA_LLM_MODEL="llama3.1:8b",
    OLLAMA_EMBED_MODEL="nomic-embed-text",
    RAG_CHUNK_MAX_CHARS=900,
    RAG_CHUNK_OVERLAP=120,
    RAG_TOP_K=5,
    RAG_MIN_SCORE=0.25,
    RAG_EMBED_BATCH_SIZE=32,
)
class DocumentAPITestCase(TestCase):
    """Test cases for the document API endpoint."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        super().setUpClass()
        # Clean up any existing test data directory
        test_data_dir = Path("/tmp/test_rag_data")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir, ignore_errors=True)

    def setUp(self):
        """Set up test fixtures."""
        self.url = reverse("document-api")
        # Reset the global RAG instance
        import rag.views as views_module

        views_module._rag = None
        # Clean up test data directory
        test_data_dir = Path("/tmp/test_rag_data")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir, ignore_errors=True)

    def test_get_empty_documents(self):
        """Test GET request returns empty list when no documents loaded."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["documents"], [])
        self.assertIn("vector_stats", data)
        self.assertIn("model_info", data)

    def test_post_clear_empty(self):
        """Test POST clear action with no documents."""
        response = self.client.post(self.url, {"action": "clear"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "cleared")
        self.assertEqual(data["documents"], [])

    def test_post_load_missing_files(self):
        """Test POST load action without files returns error."""
        response = self.client.post(self.url, {"action": "load"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("files are required", data["error"])

    def test_post_load_invalid_action(self):
        """Test POST with invalid action returns error."""
        response = self.client.post(self.url, {"action": "invalid"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("load or clear", data["error"])

    @patch("rag.views._get_rag")
    def test_post_load_valid_text_file(self, mock_get_rag):
        """Test POST load action with a valid text file."""
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ["chunk1", "chunk2"]
        mock_rag.document_names.return_value = ["test.txt"]
        mock_get_rag.return_value = mock_rag

        file_content = b"This is a test document with some content."
        file = BytesIO(file_content)
        file.name = "test.txt"
        # file.content_type = "text/plain"

        response = self.client.post(
            self.url,
            {"action": "load", "files": file},
            format="multipart",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "loaded")
        self.assertEqual(data["loaded"], ["test.txt"])
        self.assertEqual(len(data["errors"]), 0)

    @patch("rag.views._get_rag")
    def test_post_load_empty_file(self, mock_get_rag):
        """Test POST load action with an empty file."""
        mock_rag = MagicMock()
        mock_rag.add_document.side_effect = ValueError(
            "File is empty or contains no text"
        )
        mock_rag.document_names.return_value = []
        mock_get_rag.return_value = mock_rag

        file_content = b""
        file = BytesIO(file_content)
        file.name = "empty.txt"
        # file.content_type = "text/plain"

        response = self.client.post(
            self.url,
            {"action": "load", "files": file},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "failed")
        self.assertEqual(len(data["loaded"]), 0)
        self.assertEqual(len(data["errors"]), 1)

    @patch("rag.views._get_rag")
    def test_post_load_multiple_files(self, mock_get_rag):
        """Test POST load action with multiple files."""
        mock_rag = MagicMock()
        mock_rag.add_document.return_value = ["chunk1"]
        mock_rag.document_names.return_value = ["test1.txt", "test2.txt"]
        mock_get_rag.return_value = mock_rag

        file1 = BytesIO(b"Content of file 1")
        file1.name = "test1.txt"
        # file1.content_type = "text/plain"

        file2 = BytesIO(b"Content of file 2")
        file2.name = "test2.txt"
        # file2.content_type = "text/plain"

        response = self.client.post(
            self.url,
            {"action": "load", "files": [file1, file2]},
            format="multipart",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "loaded")
        self.assertEqual(len(data["loaded"]), 2)

    @patch("rag.views._get_rag")
    def test_post_clear_after_load(self, mock_get_rag):
        """Test POST clear action removes documents."""
        mock_rag = MagicMock()
        mock_rag.document_names.return_value = ["test.txt"]
        mock_get_rag.return_value = mock_rag

        # First load a document
        file = BytesIO(b"Test content")
        file.name = "test.txt"
        # file.content_type = "text/plain"

        self.client.post(
            self.url,
            {"action": "load", "files": file},
            format="multipart",
        )

        # Then clear
        mock_rag.document_names.return_value = []
        response = self.client.post(self.url, {"action": "clear"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "cleared")
        mock_rag.clear.assert_called_once()


@override_settings(
    RAG_DATA_DIR="/tmp/test_rag_data",
    OLLAMA_BASE_URL="http://127.0.0.1:11434",
    OLLAMA_LLM_MODEL="llama3.1:8b",
    OLLAMA_EMBED_MODEL="nomic-embed-text",
)
class ChatAPITestCase(TestCase):
    """Test cases for the chat API endpoint."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        super().setUpClass()
        # Clean up any existing test data directory
        test_data_dir = Path("/tmp/test_rag_data")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir, ignore_errors=True)

    def setUp(self):
        """Set up test fixtures."""
        self.url = reverse("chat-api")
        # Reset the global RAG instance
        import rag.views as views_module

        views_module._rag = None
        # Clean up test data directory
        test_data_dir = Path("/tmp/test_rag_data")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir, ignore_errors=True)

    def test_post_missing_question(self):
        """Test POST without question returns error."""
        response = self.client.post(
            self.url,
            json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("question is required", data["error"])

    def test_post_invalid_json(self):
        """Test POST with invalid JSON returns error."""
        response = self.client.post(
            self.url,
            "invalid json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("Invalid JSON", data["error"])

    def test_post_empty_question(self):
        """Test POST with empty question returns error."""
        response = self.client.post(
            self.url,
            json.dumps({"question": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("question is required", data["error"])

    @patch("rag.views._get_rag")
    def test_post_valid_question(self, mock_get_rag):
        """Test POST with valid question returns answer."""
        mock_rag = MagicMock()
        mock_rag.answer.return_value = "This is a test answer."
        mock_get_rag.return_value = mock_rag

        response = self.client.post(
            self.url,
            json.dumps({"question": "What is this about?"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], "What is this about?")
        self.assertEqual(data["answer"], "This is a test answer.")

    @patch("rag.views._get_rag")
    def test_post_ollama_error(self, mock_get_rag):
        """Test POST when Ollama service fails."""
        from rag.services import OllamaError

        mock_rag = MagicMock()
        mock_rag.answer.side_effect = OllamaError("Ollama service unavailable")
        mock_get_rag.return_value = mock_rag

        response = self.client.post(
            self.url,
            json.dumps({"question": "Test question"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("Ollama service unavailable", data["error"])

    @patch("rag.views._get_rag")
    def test_post_no_documents_loaded(self, mock_get_rag):
        """Test POST when no documents are loaded."""
        mock_rag = MagicMock()
        mock_rag.answer.return_value = "I can only respond to question on the specific of the documents no documents"
        mock_get_rag.return_value = mock_rag

        response = self.client.post(
            self.url,
            json.dumps({"question": "Test question"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("no documents", data["answer"])
