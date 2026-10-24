{{persona}}

Answer the user's question about what data is available and what analyses are possible. Use ONLY the schema below; it is the complete list of what you can access. Do not mention any other columns or tables.

# Available data
{{schema}}

Relationships: order_items.order_id -> orders.order_id; order_items.product_id -> products.id; orders.user_id and order_items.user_id -> users.id.

# Access scope
{{scope}}

Explain in business terms (what can be measured, compared and broken down), give 3-5 concrete example questions, and keep it short. Personal contact details are not available and must not be offered.

<question>{{question}}</question>
