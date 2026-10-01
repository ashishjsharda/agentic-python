"""Wikipedia as a keyless, live research source (Chapter 5)."""
import html
import re
from typing import Annotated
from urllib.parse import quote

import httpx
from pydantic import Field

from agentic.tools import tool

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = ("agentic-python-research-assistant/1.0 "
              "(https://github.com/ashishjsharda/agentic-python)")


def page_url(title):
    return ("https://en.wikipedia.org/wiki/"
            + quote(title.replace(" ", "_")))


def strip_html(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def make_wikipedia_tools(http=None):
    http = http or httpx.Client(timeout=15,
                                headers={"User-Agent": USER_AGENT})

    def api(**params):
        response = http.get(WIKIPEDIA_API,
                            params={**params, "format": "json"})
        response.raise_for_status()
        return response.json()["query"]

    @tool
    def search_wikipedia(
        query: Annotated[str, Field(description="Search terms")],
        limit: Annotated[int, Field(ge=1, le=10)] = 5,
    ):
        """Search Wikipedia. Returns page titles, URLs, and snippets.
        Read a page with read_wikipedia before citing it."""
        hits = api(action="query", list="search",
                   srsearch=query, srlimit=limit)["search"]
        return [{"title": h["title"], "url": page_url(h["title"]),
                 "snippet": strip_html(h["snippet"])} for h in hits]

    @tool
    def read_wikipedia(
        title: Annotated[str, Field(description="Exact page title")],
        max_chars: Annotated[int, Field(ge=500, le=20_000)] = 6000,
    ):
        """Read the plain text of a Wikipedia page."""
        pages = api(action="query", prop="extracts", explaintext=1,
                    redirects=1, titles=title)["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            raise LookupError(f"No Wikipedia page titled {title!r}")
        return {"title": page["title"], "url": page_url(page["title"]),
                "text": page.get("extract", "")[:max_chars]}

    return search_wikipedia, read_wikipedia
