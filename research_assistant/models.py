"""The research assistant's answer contract (Chapters 3 and 5)."""
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    url: str = Field(description="URL of a page you actually read")
    quote: str = Field(description="Short excerpt that supports the "
                                   "claim, copied from the page",
                       max_length=300)


class ResearchAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the question")
    citations: list[Citation] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    open_questions: list[str] = Field(
        default_factory=list,
        description="What you could not verify, if anything",
    )
