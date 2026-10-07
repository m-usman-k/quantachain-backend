"""In-process publish/subscribe bus.

Ingestion workers publish price ticks, whale transfers, alerts and signals here;
WebSocket endpoints and the paper-trading matcher subscribe. Each subscriber
owns a bounded queue; when a slow consumer falls behind the oldest message is
dropped rather than blocking producers. Swap for Redis Pub/Sub when the API runs
on more than one node.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class Topics:
    PRICE_TICK = "market.tick"
    CANDLE = "market.candle"
    MARKET_OVERVIEW = "market.overview"
    NEWS = "market.news"
    INGESTION_LOG = "ingestion.log"
    WHALE_TRANSFER = "onchain.whale"
    SENTIMENT = "sentiment.update"
    PREDICTION = "prediction.update"
    SIGNAL = "prediction.signal"
    FRAUD_ALERT = "fraud.alert"
    ALERT = "notification.alert"
    ORDER = "trading.order"
    PORTFOLIO = "trading.portfolio"
    SYSTEM = "system"

    ALL = "*"


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict[str, Any]
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "payload": self.payload, "published_at": self.published_at.isoformat()}


class Subscription:
    def __init__(self, bus: EventBus, topics: set[str], maxsize: int) -> None:
        self._bus = bus
        self.topics = topics
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def _offer(self, event: Event) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - race with consumer
                pass
        self.queue.put_nowait(event)

    def matches(self, topic: str) -> bool:
        return Topics.ALL in self.topics or topic in self.topics

    async def get(self, timeout: float | None = None) -> Event | None:
        if timeout is None:
            return await self.queue.get()
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except TimeoutError:
            return None

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Event]:
        while True:
            yield await self.queue.get()

    def close(self) -> None:
        self._bus.unsubscribe(self)

    async def __aenter__(self) -> Subscription:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()


class EventBus:
    def __init__(self) -> None:
        self._subscriptions: set[Subscription] = set()
        self.published = 0

    def subscribe(self, *topics: str, maxsize: int = 1000) -> Subscription:
        subscription = Subscription(self, set(topics) or {Topics.ALL}, maxsize)
        self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        self._subscriptions.discard(subscription)

    def publish(self, topic: str, payload: dict[str, Any]) -> Event:
        event = Event(topic=topic, payload=payload)
        self.published += 1
        for subscription in list(self._subscriptions):
            if subscription.matches(topic):
                subscription._offer(event)
        return event

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)


event_bus = EventBus()

__all__ = ["Event", "EventBus", "Subscription", "Topics", "event_bus"]
