"""Single source of truth about what the agent may see in thelook_ecommerce.

Allowlist (not denylist) of columns: a new PII column added upstream is hidden by default.
"""
from typing import Dict, FrozenSet, Tuple

PROJECT = "bigquery-public-data"
DATASET = "thelook_ecommerce"

SAFE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "orders": ("order_id", "user_id", "status", "gender", "created_at", "returned_at",
               "shipped_at", "delivered_at", "num_of_item"),
    "order_items": ("id", "order_id", "user_id", "product_id", "inventory_item_id", "status",
                    "created_at", "shipped_at", "delivered_at", "returned_at", "sale_price"),
    "products": ("id", "cost", "category", "name", "brand", "retail_price", "department",
                 "sku", "distribution_center_id"),
    "users": ("id", "age", "gender", "state", "country", "traffic_source", "created_at"),
}

ALLOWED_TABLES: FrozenSet[str] = frozenset(SAFE_COLUMNS)

# Used only for a clear error message + security logging; the real protection is SAFE_COLUMNS.
PII_COLUMNS: FrozenSet[str] = frozenset({
    "email", "first_name", "last_name", "street_address", "postal_code",
    "city", "latitude", "longitude", "user_geom",
})


def fq(table: str) -> str:
    return f"`{PROJECT}.{DATASET}.{table}`"
