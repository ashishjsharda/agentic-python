import os
import uuid

import pytest


@pytest.fixture
def dsn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; Postgres tests skipped")
    return url


@pytest.fixture
def namespace():
    return f"test-{uuid.uuid4().hex[:8]}"
