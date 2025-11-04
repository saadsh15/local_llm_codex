# ollama_chat/cli.py
import uuid
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from .core import (
    ollama_generate,
    print_assistant,
    print_user,
    handle_command,
)
from .db import init_db, add_message, get_history, list_sessions

app = typer.Typer(help="Colourful terminal chat with Ollama models, file I/O, search, and RAG.")
console = Console()


def choose_session() -> str:
    """Let the user pick an existing session or start a new one."""
    sessions = list_sessions()
    if not sessions:
        return str(uuid.uuid4())

    table = Table(title="Existing chat sessions")
    table.add_column("Index", justify="right")
    table.add_column("Session ID")
    table.add_column("Started")
    for idx, (sess_id, start_ts) in enumerate(sessions):
        # Format the datetime to string for Rich rendering
        table.add_row(str(idx + 1), sess_id, str(start_ts))
    console.print(table)

    choice = typer.prompt("Enter index to resume, or press ENTER for a new session", default="")
    if not choice.strip():
        return str(uuid.uuid4())
    try:
        i = int(choice) - 1
        return sessions[i][0]
    except Exception:
        console.print("[red]Invalid choice – starting a new session.[/red]")
        return str(uuid.uuid4())


@app.command()
def chat(
    model: str = typer.Option("llama3", "--model", "-m", help="Ollama model name"),
):
    """
    Start an interactive chat session.
    """
    init_db()
    session_id = choose_session()
    console.print(f"[bold]Session ID:[/bold] {session_id}")

    # Load previous history (if any) and replay it in the UI
    past = get_history(session_id)
    for role, text in past:
        if role == "assistant":
            print_assistant(text)
        else:
            print_user(text)

    # --------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------
    continue_chat = True
    while continue_chat:
        user_input = typer.prompt("\n[green]Your message[/green]")
        # --------------------------------------------------------------
        # Check if it is a slash command
        # --------------------------------------------------------------
        if user_input.strip().startswith("/"):
            should_continue, reply = handle_command(user_input, session_id)

            # If the handler returned a RAG‑prompt we need to treat it as
            # normal user input (i.e. send to model). The convention is:
            # If reply looks like a prompt (contains "**User query**"), we just
            # pass it forward.
            if reply and reply.startswith("You are a helpful assistant"):
                # It is a generated RAG prompt – treat as normal user message
                user_input = reply
                # fall‑through to normal handling
            else:
                console.print(reply)
                continue_chat = should_continue
                if not should_continue:
                    break
                else:
                    continue   # go back to ask next user input

        # --------------------------------------------------------------
        # Normal user message
        # --------------------------------------------------------------
        print_user(user_input)
        add_message(session_id, "user", user_input)  # Save user message

        # Build message list for Ollama (system messages are optional)
        hist = get_history(session_id)   # Returns List[Tuple[role, content]]
        messages = [{"role": role, "content": content} for role, content in hist]

        # Get assistant reply
        assistant_reply = ollama_generate(model, messages)
        print_assistant(assistant_reply)
        add_message(session_id, "assistant", assistant_reply)  # Save assistant reply

    # --------------------------------------------------------------
    # End of chat – optional: show session summary
    # --------------------------------------------------------------
    console.print("\n[bold magenta]Session finished.[/bold magenta]")


@app.command()
def sessions():
    """List all stored chat sessions."""
    sess = list_sessions()
    if not sess:
        console.print("[yellow]No previous sessions.[/yellow]")
        return

    table = Table(title="Saved chat sessions")
    table.add_column("Session ID")
    table.add_column("Started")
    for s_id, start in sess:
        table.add_row(s_id, str(start))
    console.print(table)


if __name__ == "__main__":
    app()
