from __future__ import annotations

import os

import uvicorn

from inference import app


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
