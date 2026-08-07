class ReplayHouseError(Exception):
    """Base class for all replayhouse errors."""


class SchemaError(ReplayHouseError):
    """Invalid table/column names, config values, or input shapes."""


class BackendError(ReplayHouseError):
    """Connection or driver-level failure."""


class TableExistsError(SchemaError):
    """create() found an existing table with the same name."""

    def __init__(self, table_name: str):
        super().__init__(f"table {table_name!r} already exists "
                         f"(pass exists_ok=True to open it instead)")
        self.table_name = table_name
