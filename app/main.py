from __future__ import annotations

import argparse
import os
import secrets

from app.frontend.ui import CSS, THEME, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Free LLM Colab Playground")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true", help="Use Gradio's temporary share link")
    args = parser.parse_args()
    should_share = args.share or os.getenv("COLAB_RELEASE_TAG") is not None
    auth = None
    if should_share:
        ui_password = os.getenv("PLAYGROUND_UI_PASSWORD") or secrets.token_urlsafe(12)
        auth = ("colab", ui_password)
        print("UI username: colab")
        print(f"UI password: {ui_password}")
    create_app().queue(default_concurrency_limit=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=should_share,
        auth=auth,
        auth_message="Username: colab. Copy the UI password printed in your launch output.",
        show_error=False,
        theme=THEME,
        css=CSS,
        max_file_size="8mb",
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
