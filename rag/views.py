import json
import os
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .services import LocalRag, OllamaError

_rag = None


def _get_rag() -> LocalRag:
    global _rag

    if _rag is None:
        data_dir = Path(
            getattr(settings, "RAG_DATA_DIR", None)
            or (Path(settings.BASE_DIR) / "rag_data")
        )

        _rag = LocalRag(
            data_dir=data_dir,
            base_url=os.environ.get(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ),
            llm_model=os.environ.get(
                "OLLAMA_LLM_MODEL",
                "llama3.1:8b",
            ),
            embed_model=os.environ.get(
                "OLLAMA_MODEL",
                "nomic-embed-text",
            ),
            chunk_max_chars=int(os.environ.get("RAG_CHUNK_MAX_CHARS", "900")),
            chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "120")),
            top_k=int(os.environ.get("RAG_TOP_K", "5")),
            min_score=float(os.environ.get("RAG_MIN_SCORE", "0.25")),
            embed_batch_size=int(os.environ.get("RAG_EMBED_BATCH_SIZE", "32")),
        )

    return _rag


def _read_text_file(file_obj) -> str:
    name = os.path.basename(file_obj.name).lower()
    content_type = (file_obj.content_type or "").lower()

    if name.endswith(".txt") or content_type.startswith("text/plain"):
        return file_obj.read().decode("utf-8", errors="ignore")

    # Future PDF support example:
    #
    # if name.endswith(".pdf"):
    #     import pypdf
    #     reader = pypdf.PdfReader(file_obj)
    #     return "\n\n".join(
    #         (page.extract_text() or "")
    #         for page in reader.pages
    #     )

    raise ValueError("Only .txt / text/plain files are supported for now.")


@csrf_exempt
@require_http_methods(["GET", "POST"])
def document_api(request):
    rag = _get_rag()

    # GET: return loaded document names
    if request.method == "GET":
        return JsonResponse(
            {
                "documents": rag.document_names(),
                "vector_stats": rag.vector_stats(),
                "model_info": rag.model_info(),
            }
        )

    # POST: load or clear
    action = (request.POST.get("action") or request.GET.get("action") or "load").lower()

    if action == "clear":
        rag.clear()
        return JsonResponse(
            {
                "status": "cleared",
                "documents": rag.document_names(),
            }
        )

    if action != "load":
        return JsonResponse(
            {"error": "action must be load or clear"},
            status=400,
        )

    files = request.FILES.getlist("files")
    if not files:
        return JsonResponse(
            {"error": "One or more files are required in the 'files' field"},
            status=400,
        )

    loaded = []
    errors = []

    for file_obj in files:
        name = str(os.path.basename(file_obj.name) or "unknown_file")

        try:
            text = _read_text_file(file_obj)
            if not text.strip():
                raise ValueError("File is empty or contains no text")

            rag.add_document(name, text)
            loaded.append(name)

        except (ValueError, OSError, UnicodeDecodeError) as exc:
            errors.append(
                {
                    "file": name,
                    "error": str(exc),
                }
            )

    status_code = 200 if loaded else 400

    return JsonResponse(
        {
            "status": "loaded" if loaded else "failed",
            "loaded": loaded,
            "errors": errors,
            "documents": rag.document_names(),
        },
        status=status_code,
    )


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)

    try:
        answer = _get_rag().answer(question)
    except OllamaError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    return JsonResponse(
        {
            "question": question,
            "answer": answer,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def export_chat_api(request):
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return JsonResponse({"error": "messages must be a list"}, status=400)

    from datetime import datetime

    from django.http import HttpResponse

    # Create a text export
    lines = []
    lines.append(
        f"Chat Export - {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("=" * 50)
    lines.append("")

    for msg in messages:
        role = "You" if msg.get("isUser") else "Assistant"
        timestamp = msg.get("timestamp", "")
        content = msg.get("content", "")
        lines.append(f"[{role}] {timestamp}")
        lines.append(content)
        lines.append("")

    response = HttpResponse("\n".join(lines), content_type="text/plain")
    filename = (
        f"chat_export_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'

    return response


def documents_page(request):
    return render(request, "rag/documents.html")


def chat_page(request):
    return render(request, "rag/chat.html")
