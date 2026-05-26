from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    PROFESSOR = "professor"
    COORDINATOR = "coordenador"


def has_role(user_role: str, allowed_roles: set[UserRole]) -> bool:
    try:
        return UserRole(user_role) in allowed_roles
    except ValueError:
        return False
