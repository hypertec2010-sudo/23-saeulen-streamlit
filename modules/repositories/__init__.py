"""Repository API introduced in v28.1."""
from modules.storage.watchlist_repository import WatchlistRepository

from .base import NamespaceRepository
from .event_repository import EventRepository
from .position_repository import PositionRepository
from .registry import RepositoryRegistry, create_repository_registry
from .trade_journal_repository import TradeJournalRepository

__all__ = [
    "NamespaceRepository",
    "PositionRepository",
    "TradeJournalRepository",
    "EventRepository",
    "WatchlistRepository",
    "RepositoryRegistry",
    "create_repository_registry",
]
