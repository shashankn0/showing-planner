import pytest

from core.db import dispose_engines


@pytest.fixture(autouse=True)
def _release_sqlite_files():
    yield
    dispose_engines()
