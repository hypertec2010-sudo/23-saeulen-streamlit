"""Central storage package introduced in v28.0."""
from .base import StorageResult
from .local_backend import LocalJsonBackend
from .manager import StorageManager, create_storage_manager, resolve_user_id, should_use_database_watchlists
from .migration import migrate_legacy_json_files
from .supabase_backend import SupabaseBackend
from .watchlist_repository import WatchlistRepository

__all__ = [
    "StorageResult",
    "LocalJsonBackend",
    "SupabaseBackend",
    "StorageManager",
    "WatchlistRepository",
    "create_storage_manager",
    "resolve_user_id",
    "should_use_database_watchlists",
    "migrate_legacy_json_files",
]
