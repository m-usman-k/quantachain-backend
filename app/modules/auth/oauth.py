"""OAuth 2.0 authorization-code flow for Google and GitHub (no extra dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, FeatureDisabledError, ValidationFailedError


@dataclass(frozen=True)
class OAuthProviderConfig:
    name: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scope: str
    extra_authorize_params: dict[str, str]


PROVIDERS: dict[str, OAuthProviderConfig] = {
    "google": OAuthProviderConfig(
        name="google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        scope="openid email profile",
        extra_authorize_params={"access_type": "online", "prompt": "select_account"},
    ),
    "github": OAuthProviderConfig(
        name="github",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        userinfo_url="https://api.github.com/user",
        scope="read:user user:email",
        extra_authorize_params={},
    ),
}


@dataclass
class OAuthProfile:
    provider: str
    provider_user_id: str
    email: str | None
    email_verified: bool
    full_name: str | None
    avatar_url: str | None


def provider_credentials(settings: Settings, provider: str) -> tuple[str, str]:
    if provider == "google" and settings.google_client_id and settings.google_client_secret:
        return settings.google_client_id, settings.google_client_secret.get_secret_value()
    if provider == "github" and settings.github_client_id and settings.github_client_secret:
        return settings.github_client_id, settings.github_client_secret.get_secret_value()
    raise FeatureDisabledError(f"OAuth provider '{provider}' is not configured")


def get_provider(provider: str) -> OAuthProviderConfig:
    config = PROVIDERS.get(provider.lower())
    if config is None:
        raise ValidationFailedError(f"Unknown OAuth provider '{provider}'")
    return config


def build_authorization_url(settings: Settings, provider: str, state: str) -> str:
    config = get_provider(provider)
    client_id, _ = provider_credentials(settings, config.name)
    params = {
        "client_id": client_id,
        "redirect_uri": settings.oauth_redirect_uri(config.name),
        "response_type": "code",
        "scope": config.scope,
        "state": state,
        **config.extra_authorize_params,
    }
    return f"{config.authorize_url}?{urlencode(params)}"


async def exchange_code(settings: Settings, provider: str, code: str) -> OAuthProfile:
    config = get_provider(provider)
    client_id, client_secret = provider_credentials(settings, config.name)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.oauth_redirect_uri(config.name),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            token_response = await client.post(config.token_url, data=data, headers={"Accept": "application/json"})
            token_response.raise_for_status()
            token_payload = token_response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(f"{config.name} token exchange failed") from exc
        access_token = token_payload.get("access_token")
        if not access_token:
            raise ExternalServiceError(f"{config.name} did not return an access token")
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        try:
            user_response = await client.get(config.userinfo_url, headers=headers)
            user_response.raise_for_status()
            info = user_response.json()
            if config.name == "github" and not info.get("email"):
                emails_response = await client.get("https://api.github.com/user/emails", headers=headers)
                if emails_response.status_code == 200:
                    info["_emails"] = emails_response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(f"{config.name} profile lookup failed") from exc

    return _normalise_profile(config.name, info)


def _normalise_profile(provider: str, info: dict) -> OAuthProfile:  # type: ignore[type-arg]
    if provider == "google":
        return OAuthProfile(
            provider=provider,
            provider_user_id=str(info.get("sub")),
            email=(info.get("email") or "").lower() or None,
            email_verified=bool(info.get("email_verified", False)),
            full_name=info.get("name"),
            avatar_url=info.get("picture"),
        )
    email = info.get("email")
    verified = bool(email)
    for entry in info.get("_emails", []) or []:
        if entry.get("primary") and entry.get("verified"):
            email, verified = entry.get("email"), True
            break
    return OAuthProfile(
        provider=provider,
        provider_user_id=str(info.get("id")),
        email=(email or "").lower() or None,
        email_verified=verified,
        full_name=info.get("name") or info.get("login"),
        avatar_url=info.get("avatar_url"),
    )


__all__ = ["PROVIDERS", "OAuthProfile", "build_authorization_url", "exchange_code", "get_provider"]
