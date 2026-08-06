import uuid

from replayhouse._ids import uuid7


def test_uuid7_is_valid_uuid_version_7():
    u = uuid.UUID(uuid7())
    assert u.version == 7


def test_uuid7_is_unique():
    ids = [uuid7() for _ in range(1000)]
    assert len(set(ids)) == 1000
