"""Launch the optional Streamlit interface."""

import sys
from importlib.util import find_spec
from pathlib import Path


def launch_ui(host="127.0.0.1", port=8501):
    if find_spec("streamlit") is None:
        raise RuntimeError('The UI dependency is missing. Install it with: pip install "GEDICorrect[ui]"')

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("ui.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return streamlit_cli.main()
