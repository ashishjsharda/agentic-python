"""Chapter 7: research over your own documents."""
from pathlib import Path

from agentic.rag import (RETRIEVE_DOCS_SCHEMA, index_document,
                         make_retrieve_docs)
from agentic.tools import Tool


def index_directory(store, root, pattern="**/*.md"):
    root = Path(root)
    count = 0
    for path in sorted(root.glob(pattern)):
        index_document(store, str(path.relative_to(root)),
                       path.read_text())
        count += 1
    return count


def make_documents_tool(store):
    return Tool(make_retrieve_docs(store), RETRIEVE_DOCS_SCHEMA)
