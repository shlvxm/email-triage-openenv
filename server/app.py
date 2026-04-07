from __future__ import annotations

import os

from inference import app, run_server


def main() -> None:
    os.environ.setdefault("PORT", os.getenv("PORT", "7860"))
    raise SystemExit(run_server())


if __name__ == "__main__":
    main()