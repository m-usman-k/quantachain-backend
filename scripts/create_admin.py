"""Create or promote an administrator.

python -m scripts.create_admin --email admin@example.com --password 'Str0ngPass!'
"""

from __future__ import annotations

import argparse
import asyncio
import getpass

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.security import hash_password
from app.db.collections import ensure_schema
from app.db.mongo import mongo
from app.modules.auth.models import Role, User
from app.modules.auth.repository import AuditLogRepository, UserRepository


async def create_admin(email: str, password: str | None, full_name: str | None) -> str:
    settings = get_settings()
    db = await mongo.connect(settings)
    await ensure_schema(db, settings)
    users = UserRepository(db)
    email = email.lower().strip()
    existing = await users.get_by_email(email)
    if existing is None:
        if not password:
            raise SystemExit("A password is required to create a new account")
        user = await users.insert(
            User(
                email=email,
                full_name=full_name,
                password_hash=hash_password(password),
                role=Role.ADMIN,
                is_verified=True,
            )
        )
        action = "created"
    else:
        changes = {"role": Role.ADMIN, "is_active": True}
        if password:
            changes["password_hash"] = hash_password(password)
        if full_name:
            changes["full_name"] = full_name
        user = await users.update(existing.id or "", changes, inc={"token_version": 1}) or existing
        action = "promoted"
    await AuditLogRepository(db).append(
        action="admin.bootstrap",
        resource="user",
        resource_id=user.id,
        metadata={"email": email, "result": action, "via": "scripts.create_admin"},
    )
    await mongo.close()
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or promote a Quantachain administrator")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", help="Prompted when omitted and the account does not exist")
    parser.add_argument("--name", dest="full_name")
    args = parser.parse_args()
    configure_logging("WARNING")
    password = args.password
    if password is None:
        entered = getpass.getpass("Password (leave empty to keep the existing one): ")
        password = entered or None
    action = asyncio.run(create_admin(args.email, password, args.full_name))
    print(f"Admin {action}: {args.email}")


if __name__ == "__main__":
    main()
