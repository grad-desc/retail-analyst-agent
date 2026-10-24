"""Who is asking and what row scope they get (no heavy dependencies)."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Role(str, Enum):
    CUSTOMER = "customer"    # sees only data tied to their own orders
    EXECUTIVE = "executive"  # sees company-wide rows; every other guard still applies


@dataclass(frozen=True)
class UserPolicy:
    """Row scope for one principal. Role must come from authenticated identity, never from the
    user's chat text. Column-level protection (no PII) is identical for every role."""
    role: Role
    user_id: Optional[int] = None  # required for CUSTOMER

    def __post_init__(self) -> None:
        if self.role is Role.CUSTOMER:
            uid = self.user_id
            if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
                raise TypeError("customer policy requires a non-negative int user_id")  # no SQL smuggling

    @property
    def row_scoped(self) -> bool:
        return self.role is Role.CUSTOMER

    @classmethod
    def customer(cls, user_id: int) -> "UserPolicy":
        return cls(Role.CUSTOMER, user_id)

    @classmethod
    def executive(cls) -> "UserPolicy":
        return cls(Role.EXECUTIVE)


def subject_of(principal: dict) -> str:
    """Stable identity that owns saved reports, preferences and feedback. In production this is the
    authenticated subject from the identity provider, never text typed in the chat."""
    return principal.get("subject") or f"{principal.get('role')}:{principal.get('user_id') or 'default'}"
