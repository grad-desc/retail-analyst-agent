import pytest

from src.sql.rls import Role, UserPolicy
from src.sql.validator import SQLValidationError, validate_and_rewrite

ME = UserPolicy.customer(123)
EXEC = UserPolicy.executive()
BOTH = [ME, EXEC]  # every safety guard must hold for every role


def kind(sql, policy):
    with pytest.raises(SQLValidationError) as e:
        validate_and_rewrite(sql, policy)
    return e.value.kind


def test_every_table_is_scoped_to_the_user_and_aliases_survive():
    out = validate_and_rewrite(
        "SELECT o.order_id, SUM(oi.sale_price) s FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id GROUP BY 1", ME)
    assert out.tables == ["order_items", "orders"]
    assert out.sql.count("user_id = 123") >= 2
    assert "`bigquery-public-data.thelook_ecommerce.orders`" in out.sql
    assert out.limit_added


def test_products_limited_to_products_the_user_ordered():
    sql = validate_and_rewrite("SELECT * FROM products", ME).sql
    assert "product_id" in sql and "user_id = 123" in sql


def test_users_limited_to_self():
    assert "id = 123" in validate_and_rewrite("SELECT age FROM users", ME).sql


@pytest.mark.parametrize("bad", ["1 OR 1=1", True, -5, 1.5, None])
def test_customer_user_id_must_be_a_real_int(bad):
    with pytest.raises(TypeError):
        UserPolicy.customer(bad)
    with pytest.raises(TypeError):
        UserPolicy(Role.CUSTOMER, bad)


def test_executive_sees_all_rows_but_only_safe_columns():
    sql = validate_and_rewrite("SELECT * FROM products", EXEC).sql
    assert "user_id" not in sql and "retail_price" in sql


def test_qualified_and_backticked_names_accepted():
    validate_and_rewrite("SELECT COUNT(*) FROM `bigquery-public-data.thelook_ecommerce.products`", ME)


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "DELETE FROM orders WHERE TRUE", "INSERT INTO orders VALUES (1)",
    "UPDATE orders SET status='x' WHERE TRUE", "CREATE TABLE x AS SELECT 1",
])
@pytest.mark.parametrize("policy", BOTH)
def test_non_select_rejected(sql, policy):
    assert kind(sql, policy) in {"forbidden_statement", "syntax"}


@pytest.mark.parametrize("policy", BOTH)
def test_multi_statement_rejected(policy):
    assert kind("SELECT 1 FROM orders; DROP TABLE orders", policy) == "multi_statement"


def test_trailing_semicolon_ok():
    validate_and_rewrite("SELECT 1 FROM orders;", ME)


@pytest.mark.parametrize("sql", [
    "SELECT email FROM users", "SELECT u.first_name FROM users u",
    "SELECT COUNT(DISTINCT city) FROM users", "SELECT latitude FROM users",
])
@pytest.mark.parametrize("policy", BOTH)
def test_pii_columns_rejected(sql, policy):
    assert kind(sql, policy) == "pii_column"


@pytest.mark.parametrize("policy", BOTH)
def test_select_star_never_exposes_pii(policy):
    out = validate_and_rewrite("SELECT * FROM users", policy).sql.lower()
    for c in ("email", "first_name", "last_name", "street_address", "latitude", "city"):
        assert c not in out


@pytest.mark.parametrize("sql", [
    "SELECT * FROM `bigquery-public-data.thelook_ecommerce.events`",
    "SELECT * FROM `bigquery-public-data.thelook_ecommerce.inventory_items`",
    "SELECT * FROM `bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS`",
    "SELECT * FROM `other-project.thelook_ecommerce.orders`",
    "SELECT * FROM `bigquery-public-data.samples.shakespeare`",
    "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders*`",
])
@pytest.mark.parametrize("policy", BOTH)
def test_foreign_tables_rejected(sql, policy):
    assert kind(sql, policy) == "forbidden_table"


@pytest.mark.parametrize("policy", BOTH)
def test_cte_cannot_shadow_real_table(policy):
    assert kind("WITH orders AS (SELECT 1 AS order_id) SELECT * FROM orders", policy) == "policy"


def test_cte_reference_allowed():
    out = validate_and_rewrite(
        "WITH t AS (SELECT category, COUNT(*) c FROM products GROUP BY 1) SELECT * FROM t", ME)
    assert out.tables == ["products"]


@pytest.mark.parametrize("policy", BOTH)
def test_dangerous_functions_rejected(policy):
    ok = {"forbidden_table", "forbidden_function"}
    assert kind("SELECT * FROM EXTERNAL_QUERY('x','select 1')", policy) in ok
    assert kind("SELECT * FROM ML.PREDICT(MODEL m, (SELECT 1))", policy) in ok


def test_existing_limit_kept():
    assert not validate_and_rewrite("SELECT * FROM orders LIMIT 5", ME).limit_added


def test_syntax_error_has_llm_feedback():
    with pytest.raises(SQLValidationError) as e:
        validate_and_rewrite("SELEC FROM", ME)
    assert e.value.feedback.startswith("[syntax]")
