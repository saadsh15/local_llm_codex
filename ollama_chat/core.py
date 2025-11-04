# ollama_chat/core.py
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Tuple

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .db import init_db, add_message, get_history
from .file_ops import read_file, write_file
from .websearch import search
from .rag import add_documents, retrieve

# ----------------------------------------------------------------------
# Ollama API wrapper (modified for /api/generate endpoint)
# ----------------------------------------------------------------------
OLLAMA_HOST = "http://127.0.0.1:11434"

def ollama_generate(model: str, messages: List[dict]) -> str:
    """
    Calls Ollama's /api/generate endpoint (for older versions like 0.12.9).
    Converts the messages list to a single prompt string.
    Returns the assistant's response as plain text.
    """
    # Combine all messages into a single prompt
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    prompt = "\n".join(prompt_parts) + "\nAssistant:"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    with httpx.Client(base_url=OLLAMA_HOST, timeout=600.0) as client:
        resp = client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
console = Console()

def print_assistant(text: str) -> None:
    md = Markdown(text, code_theme="monokai")
    panel = Panel(md, title="[bold cyan]Assistant[/bold cyan]", border_style="cyan")
    console.print(panel)

def print_user(text: str) -> None:
    panel = Panel(text, title="[bold green]You[/bold green]", border_style="green")
    console.print(panel)

# ----------------------------------------------------------------------
# Command parsing
# ----------------------------------------------------------------------
def handle_command(command: str, session_id: str) -> Tuple[bool, str]:
    """
    Returns (should_continue, optional_reply_to_user).
    If should_continue is False the chat terminates.
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    # ------------------------------------------------------------------
    # /exit
    # ------------------------------------------------------------------
    if cmd in {"/exit", "/quit", "/q"}:
        return False, "Bye! 👋"

    # ------------------------------------------------------------------
    # /file read <path>
    # ------------------------------------------------------------------
    if cmd == "/file" and arg.startswith("read"):
        try:
            _, path = arg.split(maxsplit=1)
            content = read_file(path)
            add_documents(session_id, [content])   # remember for RAG
            return True, f"[file read] {path}:\n{content}"
        except Exception as exc:
            return True, f"[file read error] {exc}"

    # ------------------------------------------------------------------
    # /file write <path> <content>
    # ------------------------------------------------------------------
    if cmd == "/file" and arg.startswith("write"):
        try:
            # /file write <path> <content>
            _, rest = arg.split(maxsplit=1)
            path, content = rest.split(maxsplit=1)
            write_file(path, content, overwrite=True)
            add_documents(session_id, [content])
            return True, f"[file written] {path}"
        except Exception as exc:
            return True, f"[file write error] {exc}"

    # ------------------------------------------------------------------
    # /search <query>
    # ------------------------------------------------------------------
    if cmd == "/search":
        try:
            results = search(arg, max_results=3)
            formatted = "\n".join(f"* **{t}** – {s}" for t, s in results)
            return True, f"[search results]\n{formatted}"
        except Exception as exc:
            return True, f"[search error] {exc}"

    # ------------------------------------------------------------------
    # /rag <query>
    # ------------------------------------------------------------------
    if cmd == "/rag":
        try:
            context = retrieve(session_id, arg)
            if not context:
                return True, "[RAG] No knowledge yet – try adding files or web‑search first."
            # Send the query + retrieved context to the model
            rag_prompt = f"""You are a helpful assistant.

**User query:** {arg}

**Relevant knowledge:**
{context}

Answer the user using only the knowledge above. If you cannot answer, say you don't know."""
            return True, rag_prompt  # The caller will treat it as normal user input
        except Exception as exc:
            return True, f"[RAG error] {exc}"

    # ------------------------------------------------------------------
    # Unknown command
    # ------------------------------------------------------------------
    return True, f"[unknown command] {command}"
