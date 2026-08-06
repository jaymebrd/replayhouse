import os
import uuid

import pytest

from replayhouse.store import ReplayHouse

URL = os.environ.get("REPLAYHOUSE_TEST_URL")


@pytest.fixture
def server_store():
    if not URL:
        pytest.skip("set REPLAYHOUSE_TEST_URL to run integration tests")
    s = ReplayHouse.connect(URL)
    yield s
    s.close()


@pytest.fixture
def fresh_name(server_store):
    name = f"rh_it_{uuid.uuid4().hex[:8]}"
    yield name
    server_store.drop(name)
