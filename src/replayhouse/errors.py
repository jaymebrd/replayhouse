class ReplayHouseError(Exception):
    """Base class for all replayhouse errors."""


class SchemaError(ReplayHouseError):
    """Invalid table/column names, config values, or input shapes."""


class BackendError(ReplayHouseError):
    """Connection or driver-level failure."""
