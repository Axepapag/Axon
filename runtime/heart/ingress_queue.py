"""Heart-owned ingress queue.

External arrivals (user, tool, advisor) land here first.  While a cognitive
 tick is in flight, items remain queued for the next heartbeat.  Between ticks,
 the heart drains the queue and commits each item as a typed delta through the
 heart transaction boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from .authority import IngressChannel


@dataclass(frozen=True, slots=True)
class IngressItem:
    """One external arrival waiting to cross the heart boundary."""

    channel: IngressChannel
    text: str
    provenance: str
    enqueued_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.channel, IngressChannel):
            raise TypeError("IngressItem.channel must be an IngressChannel")
        if not isinstance(self.text, str):
            raise TypeError("IngressItem.text must be a string")
        if not isinstance(self.provenance, str):
            raise TypeError("IngressItem.provenance must be a string")
        if not isinstance(self.enqueued_at, datetime):
            raise TypeError("IngressItem.enqueued_at must be a datetime")


class IngressQueue:
    """In-memory FIFO for heart-governed external arrivals."""

    def __init__(self) -> None:
        self._items: list[IngressItem] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        return not self._items

    def enqueue(
        self,
        channel: IngressChannel,
        text: str,
        *,
        provenance: str = "",
    ) -> IngressItem:
        """Append one arrival to the queue."""

        item = IngressItem(
            channel=channel,
            text=text,
            provenance=provenance,
            enqueued_at=datetime.now(timezone.utc),
        )
        self._items.append(item)
        return item

    def peek(self, count: int = 1) -> tuple[IngressItem, ...]:
        """Return up to ``count`` items from the front without removing them."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("IngressQueue.peek count must be an integer")
        if count < 0:
            raise ValueError("IngressQueue.peek count must be non-negative")
        return tuple(self._items[:count])

    def acknowledge(self, count: int = 1) -> None:
        """Remove ``count`` items from the front after successful processing."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("IngressQueue.acknowledge count must be an integer")
        if count < 0:
            raise ValueError("IngressQueue.acknowledge count must be non-negative")
        if count > len(self._items):
            raise ValueError(
                f"cannot acknowledge {count} items; only {len(self._items)} queued"
            )
        del self._items[:count]

    def drain(self) -> tuple[IngressItem, ...]:
        """Atomically remove and return all queued items in arrival order.

        Prefer :meth:`peek` + :meth:`acknowledge` for transactional processing.
        """

        items = tuple(self._items)
        self._items.clear()
        return items

    def __iter__(self) -> Iterator[IngressItem]:
        return iter(self._items)


__all__ = [
    "IngressItem",
    "IngressQueue",
]
