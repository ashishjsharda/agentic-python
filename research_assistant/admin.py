"""Chapter 18: operator commands.

    python -m research_assistant.admin create-key alice
    python -m research_assistant.admin index-docs ./docs
"""
import os
import sys

from agentic.pgmemory import PgVectorStore

from .db import Database
from .documents import index_directory


def main(argv):
    dsn = os.environ["DATABASE_URL"]
    command, arg = argv[1], argv[2]
    if command == "create-key":
        print(Database(dsn).create_api_key(arg))
    elif command == "index-docs":
        store = PgVectorStore(dsn, namespace="documents")
        print(f"Indexed {index_directory(store, arg)} documents")
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main(sys.argv)
