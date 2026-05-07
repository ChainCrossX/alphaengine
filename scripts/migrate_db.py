"""
One-shot DB initializer. Reads DATABASE_URL from env, creates tables.

Usage:
    python scripts/migrate_db.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from alphaengine.db.models import init_db


def main() -> int:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set in environment or .env")
        return 1
    print(f"Initializing schema on {url.split('@')[-1]}")
    init_db(url)
    print("Done. Tables created if they did not already exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
