"""News sources: NewsAPI (keyed), RSS feeds and Reddit (public JSON)."""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import structlog

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.timeutils import utcnow
from app.integrations.base import ProviderClient

logger = structlog.get_logger(__name__)

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "ref"}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class RawArticle:
    url: str
    title: str
    source_name: str
    origin: str  # newsapi | rss
    published_at: datetime
    summary: str | None = None
    content: str | None = None
    author: str | None = None
    image_url: str | None = None
    language: str = "en"


@dataclass(slots=True)
class RawSocialPost:
    platform: str
    external_id: str
    title: str
    text: str
    url: str
    author: str | None
    posted_at: datetime
    score: int = 0
    comments: int = 0
    community: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------- helpers
def sanitize_text(value: str | None, *, max_length: int = 5000) -> str | None:
    """Strip HTML tags, unescape entities and collapse whitespace."""
    if not value:
        return None
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_length] or None


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower(), path, urlencode(query), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def content_hash(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9 ]", "", text.lower())
    normalised = _WS_RE.sub(" ", normalised).strip()
    return hashlib.sha256(normalised.encode()).hexdigest()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


# ----------------------------------------------------------------- NewsAPI
class NewsApiClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        headers = {}
        if settings.newsapi_key:
            headers["X-Api-Key"] = settings.newsapi_key.get_secret_value()
        super().__init__("newsapi", settings.newsapi_url, headers=headers, timeout=20.0)
        self.configured = settings.newsapi_key is not None

    async def everything(
        self, query: str, *, since: datetime | None = None, page_size: int = 50, language: str = "en"
    ) -> list[RawArticle]:
        if not self.configured:
            return []
        params: dict[str, Any] = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100),
        }
        if since is not None:
            params["from"] = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
        payload = await self.get_json("/everything", params=params)
        if payload.get("status") != "ok":
            raise ExternalServiceError(f"NewsAPI error: {payload.get('message', 'unknown')}")
        articles: list[RawArticle] = []
        for item in payload.get("articles", []):
            url = item.get("url")
            title = sanitize_text(item.get("title"), max_length=300)
            published = _parse_datetime(item.get("publishedAt"))
            if not url or not title or published is None or title == "[Removed]":
                continue
            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    source_name=(item.get("source") or {}).get("name") or "NewsAPI",
                    origin="newsapi",
                    published_at=published,
                    summary=sanitize_text(item.get("description"), max_length=1000),
                    content=sanitize_text(item.get("content")),
                    author=sanitize_text(item.get("author"), max_length=200),
                    image_url=item.get("urlToImage"),
                    language=language,
                )
            )
        return articles


# --------------------------------------------------------------------- RSS
class RssClient(ProviderClient):
    def __init__(self) -> None:
        super().__init__("rss", timeout=20.0, retries=1)

    async def fetch(self, feed_url: str) -> list[RawArticle]:
        response = await self.request("GET", feed_url)
        parsed = await asyncio.to_thread(feedparser.parse, response.content)
        source_name = sanitize_text(parsed.feed.get("title"), max_length=120) or urlsplit(feed_url).netloc
        articles: list[RawArticle] = []
        for entry in parsed.entries:
            link = entry.get("link")
            title = sanitize_text(entry.get("title"), max_length=300)
            if not link or not title:
                continue
            struct = entry.get("published_parsed") or entry.get("updated_parsed")
            published = datetime.fromtimestamp(calendar.timegm(struct), tz=UTC) if struct else utcnow()
            summary = entry.get("summary") or (entry.get("content") or [{}])[0].get("value")
            image = None
            for media in entry.get("media_content", []) or []:
                if media.get("url"):
                    image = media["url"]
                    break
            articles.append(
                RawArticle(
                    url=link,
                    title=title,
                    source_name=source_name,
                    origin="rss",
                    published_at=published,
                    summary=sanitize_text(summary, max_length=1000),
                    author=sanitize_text(entry.get("author"), max_length=200),
                    image_url=image,
                )
            )
        return articles


# ------------------------------------------------------------------ Reddit
class RedditClient(ProviderClient):
    def __init__(self) -> None:
        super().__init__(
            "reddit",
            "https://www.reddit.com",
            headers={"User-Agent": "quantachain-research-bot/0.1 (academic project)"},
            timeout=20.0,
            retries=1,
        )

    async def new_posts(self, subreddit: str, *, limit: int = 50) -> list[RawSocialPost]:
        payload = await self.get_json(f"/r/{subreddit}/new.json", params={"limit": min(limit, 100)})
        posts: list[RawSocialPost] = []
        for child in payload.get("data", {}).get("children", []):
            data = child.get("data", {})
            if not data.get("id") or data.get("stickied"):
                continue
            title = sanitize_text(data.get("title"), max_length=300) or ""
            text = sanitize_text(data.get("selftext"), max_length=3000) or ""
            posts.append(
                RawSocialPost(
                    platform="reddit",
                    external_id=f"reddit:{data['id']}",
                    title=title,
                    text=text,
                    url=f"https://www.reddit.com{data.get('permalink', '')}",
                    author=data.get("author"),
                    posted_at=datetime.fromtimestamp(float(data.get("created_utc", 0)), tz=UTC),
                    score=int(data.get("score", 0)),
                    comments=int(data.get("num_comments", 0)),
                    community=subreddit,
                    extra={"upvote_ratio": data.get("upvote_ratio"), "flair": data.get("link_flair_text")},
                )
            )
        return posts


__all__ = [
    "NewsApiClient",
    "RawArticle",
    "RawSocialPost",
    "RedditClient",
    "RssClient",
    "canonical_url",
    "content_hash",
    "sanitize_text",
    "url_hash",
]
