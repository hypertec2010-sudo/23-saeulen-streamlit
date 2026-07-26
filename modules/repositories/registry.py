"""Central repository registry for dependency injection."""
from __future__ import annotations

from dataclasses import dataclass

from .event_repository import EventRepository
from .position_repository import PositionRepository
from .trade_journal_repository import TradeJournalRepository


@dataclass(slots=True)
class RepositoryRegistry:
    positions: PositionRepository
    trade_journal: TradeJournalRepository
    events: EventRepository


def create_repository_registry(storage) -> RepositoryRegistry:
    return RepositoryRegistry(
        positions=PositionRepository(storage),
        trade_journal=TradeJournalRepository(storage),
        events=EventRepository(storage),
    )
