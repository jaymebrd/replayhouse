from __future__ import annotations

import re

from .ddl import sidecar_name, validate_name
from .errors import SchemaError

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_PRIORITY_WORD = re.compile(r"\bpriority\b")

_CURRENT_CTE = (
    "WITH current AS\n(\n"
    "    SELECT id, argMax(priority, version) AS priority\n"
    "    FROM `{sidecar}`\n    GROUP BY id\n)\n"
)


def sample_key(by: str, seed: int | None = None, id_expr: str = "id") -> str:
    if seed is None:
        u = "1 - randCanonical()"
    else:
        u = f"(cityHash64({id_expr}, {int(seed)}) + 1) / 18446744073709551616."
    return f"-log({u}) / ({by})"


def phase1_sql(name: str, k: int, by: str = "priority", where: str | None = None, seed: int | None = None) -> str:
    validate_name(name)
    k = int(k)
    sidecar = sidecar_name(name)
    if by.strip() == "priority":
        filt = f"\n  AND id IN (SELECT id FROM `{name}` WHERE ({where}))" if where else ""
        return (
            _CURRENT_CTE.format(sidecar=sidecar)
            + f"SELECT id\nFROM current\nWHERE priority > 0{filt}\n"
            + f"ORDER BY {sample_key('priority', seed=seed)} ASC\nLIMIT {k}"
        )
    cond = f"(({by})) > 0" + (f" AND (({where}))" if where else "")
    order = f"ORDER BY {sample_key(f'({by})', seed=seed, id_expr='m.id')} ASC\nLIMIT {k}"
    if _PRIORITY_WORD.search(by):
        return (
            _CURRENT_CTE.format(sidecar=sidecar)
            + f"SELECT m.id AS id\nFROM `{name}` AS m\n"
            + f"INNER JOIN current AS c ON m.id = c.id\nWHERE {cond}\n{order}"
        )
    order = f"ORDER BY {sample_key(f'({by})', seed=seed)} ASC\nLIMIT {k}"
    return f"SELECT id\nFROM `{name}`\nWHERE {cond}\n{order}"


def validate_ids(ids: list[str]) -> list[str]:
    for i in ids:
        if not _UUID_RE.match(str(i)):
            raise SchemaError(f"not a UUID: {i!r}")
    return list(ids)


def stratified_sql(name: str, k: int, per_group: int, stratify_by: str,
                   by: str = "priority", where: str | None = None, seed: int | None = None) -> str:
    validate_name(name)
    validate_name(stratify_by)
    k, per_group = int(k), int(per_group)
    cond = f"(({by})) > 0" + (f" AND (({where}))" if where else "")
    order = f"ORDER BY strat ASC, {sample_key(f'({by})', seed=seed, id_expr='m.id')} ASC\n    LIMIT {per_group} BY strat"
    if _PRIORITY_WORD.search(by):
        inner = (
            f"    SELECT m.id AS id, m.`{stratify_by}` AS strat\n"
            f"    FROM `{name}` AS m\n"
            f"    INNER JOIN current AS c ON m.id = c.id\n"
            f"    WHERE {cond}\n    {order}"
        )
        return (
            _CURRENT_CTE.format(sidecar=sidecar_name(name))
            + f"SELECT id FROM\n(\n{inner}\n)\nLIMIT {k}"
        )
    order = f"ORDER BY strat ASC, {sample_key(f'({by})', seed=seed)} ASC\n    LIMIT {per_group} BY strat"
    inner = (
        f"    SELECT id, `{stratify_by}` AS strat\n"
        f"    FROM `{name}`\n    WHERE {cond}\n    {order}"
    )
    return f"SELECT id FROM\n(\n{inner}\n)\nLIMIT {k}"


def fetch_sql(name: str, ids: list[str]) -> str:
    validate_name(name)
    id_list = ", ".join(f"'{i}'" for i in validate_ids(ids))
    return f"SELECT * FROM `{name}` WHERE id IN ({id_list})"
