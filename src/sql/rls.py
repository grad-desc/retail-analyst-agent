"""Row-level security + column projection, applied to the SQL AST (never via the prompt).

A user is a customer of thelook_ecommerce (users.id). "Data related to him" is derived from the
data itself, no permission config needed:

    orders, order_items : rows where user_id = <me>
    products            : products that appear in MY order_items (order_items.product_id)
    users               : my own row

Executives see company-wide rows (no row filter).

Every base-table reference is replaced by a subquery that (1) projects only SAFE_COLUMNS, so
`SELECT *` can never leak PII, for EVERY role, and (2) applies the row filter above for customers.
"""
from sqlglot import exp, parse_one

from .policy import Role, UserPolicy  # noqa: F401 (re-exported)
from .schema import SAFE_COLUMNS, fq


def scoped_select(table: str, policy: UserPolicy) -> str:
    base = f"SELECT {', '.join(SAFE_COLUMNS[table])} FROM {fq(table)}"
    if not policy.row_scoped:
        return base  # executive: all rows, but still only SAFE_COLUMNS (no PII)
    uid = int(policy.user_id)
    if table in ("orders", "order_items"):
        return f"{base} WHERE user_id = {uid}"
    if table == "users":
        return f"{base} WHERE id = {uid}"
    if table == "products":
        return (f"{base} WHERE id IN "
                f"(SELECT product_id FROM {fq('order_items')} WHERE user_id = {uid})")
    raise ValueError(f"no RLS rule for table {table}")


def apply_rls(tree: exp.Expression, tables: list, policy: UserPolicy) -> exp.Expression:
    """Replace the given (already validated) exp.Table nodes with scoped subqueries.

    `tables` is collected BEFORE rewriting, so injected subqueries are never re-processed.
    """
    for node in tables:
        name = node.name.lower()
        alias = node.alias or node.name
        sub = exp.Subquery(
            this=parse_one(scoped_select(name, policy), read="bigquery"),
            alias=exp.TableAlias(this=exp.to_identifier(alias)),
        )
        if node is tree:
            tree = sub
        else:
            node.replace(sub)
    return tree
