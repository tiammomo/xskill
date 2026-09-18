"""Isolate SQLite lifecycle calls that can deadlock with CPython's GIL.

CPython releases the GIL in parts of SQLite's Unix VFS and shared-memory
code.  A second thread can otherwise finalize a connection while holding the
GIL, wait for a SQLite mutex, and deadlock with the first thread as it waits to
reacquire the GIL.  Connections created here isolate connection finalization
from ordinary SQLite operations.  Connection opens, execute, fetch, cursor
finalization, commit and rollback remain concurrent so the control plane can
handle high request load and a cursor cannot delay its own transaction's
commit.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator
from typing import Any, TypeVar


_ConnectionT = TypeVar("_ConnectionT")
logger = logging.getLogger("xskill.sqlite")

try:  # api.app prefers this newer SQLite build when it is installed.
    import pysqlite3 as _pysqlite3
except ImportError:  # pragma: no cover - optional runtime dependency
    _pysqlite3 = None


class _SQLiteCallGate:
    """Allow SQLite calls concurrently but isolate connection finalization.

    CPython releases the GIL while executing and fetching rows, so those calls
    may safely overlap.  Connection finalization can wait for SQLite's process
    mutex while retaining the GIL; it takes the exclusive side of this gate and
    cannot overlap any active SQLite call.  Existing connections may continue
    into ``commit()`` or ``rollback()`` while finalization waits; otherwise an
    SQL writer waiting on another transaction could prevent that transaction
    from releasing its database lock.  A recursive SQLite callback may also
    re-enter the shared side on its current thread.

    Waiting finalization holds new SQL calls back, so a steadily busy process
    cannot starve it, while calls that already hold the shared side keep their
    re-entry rights and cannot deadlock against the finalization they precede.
    Calls on a connection inside an open transaction pass a waiting finalization
    too: it already holds database locks, and holding it back would keep those
    locks from the running SQL call the finalization is itself waiting for.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._reader_depth: dict[int, int] = {}
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._finalizers_waiting = 0

    def enter(
        self, *, exclusive: bool, holds_database_lock: bool = False,
    ) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if exclusive:
                if self._writer_thread == thread_id:
                    self._writer_depth += 1
                    return
                if self._reader_depth.get(thread_id, 0):
                    raise RuntimeError(
                        "cannot finalize SQLite from inside an active SQL call"
                    )
                self._finalizers_waiting += 1
                try:
                    while self._writer_thread is not None or self._readers:
                        self._condition.wait()
                except BaseException:
                    self._finalizers_waiting -= 1
                    self._condition.notify_all()
                    raise
                self._finalizers_waiting -= 1
                self._writer_thread = thread_id
                self._writer_depth = 1
                return

            reader_depth = self._reader_depth.get(thread_id, 0)
            # Finalization runs SQLite teardown that can re-enter this side on
            # its own thread; both re-entries own the gate already.
            if not reader_depth and self._writer_thread != thread_id:
                while self._writer_thread is not None or (
                    self._finalizers_waiting and not holds_database_lock
                ):
                    self._condition.wait()
            self._readers += 1
            self._reader_depth[thread_id] = reader_depth + 1

    def leave(self, *, exclusive: bool) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if exclusive:
                if self._writer_thread != thread_id or self._writer_depth <= 0:
                    raise RuntimeError("SQLite exclusive gate released by a non-owner")
                self._writer_depth -= 1
                if self._writer_depth == 0:
                    self._writer_thread = None
                    self._condition.notify_all()
                return

            depth = self._reader_depth.get(thread_id, 0)
            if depth <= 0 or self._readers <= 0:
                raise RuntimeError("SQLite shared gate released by a non-owner")
            self._readers -= 1
            if depth == 1:
                self._reader_depth.pop(thread_id, None)
            else:
                self._reader_depth[thread_id] = depth - 1
            if self._readers == 0:
                self._condition.notify_all()




class _LockedIterator(Iterator[Any]):
    """Guard lazy SQLite-backed iterators such as ``Connection.iterdump``."""

    def __init__(self, connection: "_LockedConnection", iterator: Iterator[Any]):
        self._connection = connection
        self._iterator = iterator

    def __iter__(self) -> "_LockedIterator":
        return self

    def __next__(self) -> Any:
        return self._connection._sqlite_call(lambda: next(self._iterator))


class _LockedBlob:
    """Proxy an incremental BLOB handle through its owning connection."""

    def __init__(self, connection: "_LockedConnection", blob: Any):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_blob", blob)

    def _call(
        self, operation: Callable[[], Any], *, exclusive: bool = False,
    ) -> Any:
        return self._connection._sqlite_call(operation, exclusive=exclusive)

    def __getattr__(self, name: str) -> Any:
        attribute = self._call(lambda: getattr(self._blob, name))
        if not callable(attribute):
            return attribute

        def guarded(*args: Any, **kwargs: Any) -> Any:
            return self._call(lambda: attribute(*args, **kwargs))

        return guarded

    def __len__(self) -> int:
        return self._call(lambda: len(self._blob))

    def __getitem__(self, key: Any) -> Any:
        return self._call(lambda: self._blob[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        self._call(lambda: self._blob.__setitem__(key, value))

    def __enter__(self) -> "_LockedBlob":
        self._call(self._blob.__enter__)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._call(
            lambda: self._blob.__exit__(exc_type, exc, traceback)
        )

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        blob = getattr(self, "_blob", None)
        if connection is None or blob is None:
            return
        try:
            connection._sqlite_call(blob.close)
        except Exception:
            logger.debug("failed to finalize SQLite blob", exc_info=True)


class _LockedCursor(sqlite3.Cursor):
    """Cursor subclass that routes execution and result reads through a lock."""

    def _call(
        self, operation: Callable[[], Any], *, exclusive: bool = False,
    ) -> Any:
        connection = sqlite3.Cursor.connection.__get__(self, type(self))
        return connection._sqlite_call(operation, exclusive=exclusive)

    def execute(self, *args: Any, **kwargs: Any) -> "_LockedCursor":
        self._call(lambda: sqlite3.Cursor.execute(self, *args, **kwargs))
        return self

    def executemany(self, *args: Any, **kwargs: Any) -> "_LockedCursor":
        self._call(lambda: sqlite3.Cursor.executemany(self, *args, **kwargs))
        return self

    def executescript(self, *args: Any, **kwargs: Any) -> "_LockedCursor":
        self._call(lambda: sqlite3.Cursor.executescript(self, *args, **kwargs))
        return self

    def fetchone(self) -> Any:
        return self._call(lambda: sqlite3.Cursor.fetchone(self))

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(lambda: sqlite3.Cursor.fetchmany(self, *args, **kwargs))

    def fetchall(self) -> Any:
        return self._call(lambda: sqlite3.Cursor.fetchall(self))

    def close(self) -> Any:
        return self._call(lambda: sqlite3.Cursor.close(self))

    def __iter__(self) -> "_LockedCursor":
        return self

    def __next__(self) -> Any:
        return self._call(lambda: sqlite3.Cursor.__next__(self))

    def __del__(self) -> None:
        try:
            connection = sqlite3.Cursor.connection.__get__(self, type(self))
            if getattr(connection, "_sqlite_closed", False):
                return
            self._call(lambda: sqlite3.Cursor.close(self))
        except Exception:
            logger.debug("failed to finalize SQLite cursor", exc_info=True)


class _LockedConnection(sqlite3.Connection):
    """Connection subclass separating SQL work from lifecycle operations."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._sqlite_closed = False
        self._sqlite_gate = _SQLiteCallGate()

    def _sqlite_in_transaction(self) -> bool:
        try:
            return bool(
                sqlite3.Connection.in_transaction.__get__(self, type(self))
            )
        except Exception:  # pragma: no cover - closed or half-built handle
            return False

    def _sqlite_call(
        self, operation: Callable[[], Any], *, exclusive: bool = False,
    ) -> Any:
        """Run one SQLite call under the shared or lifecycle-exclusive gate."""
        self._sqlite_gate.enter(
            exclusive=exclusive,
            holds_database_lock=not exclusive and self._sqlite_in_transaction(),
        )
        try:
            return operation()
        finally:
            self._sqlite_gate.leave(exclusive=exclusive)

    def cursor(self, *args: Any, **kwargs: Any) -> _LockedCursor:
        factory = kwargs.pop("factory", _LockedCursor)
        return self._sqlite_call(
            lambda: sqlite3.Connection.cursor(
                self, *args, factory=factory, **kwargs,
            )
        )

    def _statement(self, method: Callable[..., Any], *args: Any, **kwargs: Any):
        def run():
            cursor = sqlite3.Connection.cursor(self, factory=_LockedCursor)
            try:
                method(cursor, *args, **kwargs)
            except BaseException:
                sqlite3.Cursor.close(cursor)
                raise
            return cursor

        return self._sqlite_call(run)

    def execute(self, *args: Any, **kwargs: Any) -> _LockedCursor:
        return self._statement(sqlite3.Cursor.execute, *args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> _LockedCursor:
        return self._statement(sqlite3.Cursor.executemany, *args, **kwargs)

    def executescript(self, *args: Any, **kwargs: Any) -> _LockedCursor:
        return self._statement(sqlite3.Cursor.executescript, *args, **kwargs)

    def commit(self) -> Any:
        return self._sqlite_call(lambda: sqlite3.Connection.commit(self))

    def rollback(self) -> Any:
        return self._sqlite_call(lambda: sqlite3.Connection.rollback(self))

    def close(self) -> Any:
        if self._sqlite_closed:
            return None

        def close_once() -> Any:
            if self._sqlite_closed:
                return None
            result = sqlite3.Connection.close(self)
            self._sqlite_closed = True
            return result

        return self._sqlite_call(
            close_once, exclusive=True,
        )

    def backup(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(target, _SerializedConnection):
            target = target._connection
        return self._sqlite_call(
            lambda: sqlite3.Connection.backup(self, target, *args, **kwargs)
        )

    def iterdump(self, *args: Any, **kwargs: Any) -> _LockedIterator:
        iterator = self._sqlite_call(
            lambda: sqlite3.Connection.iterdump(self, *args, **kwargs)
        )
        return _LockedIterator(self, iterator)

    def blobopen(self, *args: Any, **kwargs: Any) -> _LockedBlob:
        blob = self._sqlite_call(
            lambda: sqlite3.Connection.blobopen(self, *args, **kwargs)
        )
        return _LockedBlob(self, blob)

    def create_aggregate(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.create_aggregate(self, *args, **kwargs)
        )

    def create_collation(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.create_collation(self, *args, **kwargs)
        )

    def create_function(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.create_function(self, *args, **kwargs)
        )

    def create_window_function(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.create_window_function(
                self, *args, **kwargs,
            )
        )

    def enable_load_extension(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.enable_load_extension(
                self, *args, **kwargs,
            )
        )

    def load_extension(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.load_extension(self, *args, **kwargs)
        )

    def set_authorizer(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.set_authorizer(self, *args, **kwargs)
        )

    def set_progress_handler(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.set_progress_handler(
                self, *args, **kwargs,
            )
        )

    def set_trace_callback(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.set_trace_callback(
                self, *args, **kwargs,
            )
        )

    def getlimit(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.getlimit(self, *args, **kwargs)
        )

    def setlimit(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.setlimit(self, *args, **kwargs)
        )

    def serialize(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.serialize(self, *args, **kwargs)
        )

    def deserialize(self, *args: Any, **kwargs: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.deserialize(self, *args, **kwargs)
        )

    def __enter__(self) -> "_LockedConnection":
        self._sqlite_call(lambda: sqlite3.Connection.__enter__(self))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._sqlite_call(
            lambda: sqlite3.Connection.__exit__(self, exc_type, exc, traceback)
        )

    def interrupt(self) -> Any:
        # SQLite explicitly supports this call from another thread.  Waiting
        # for the operation lock would make it unable to interrupt that call.
        return sqlite3.Connection.interrupt(self)

    def __del__(self) -> None:
        if getattr(self, "_sqlite_closed", False):
            return
        try:
            self.close()
        except Exception:
            logger.debug("failed to finalize SQLite connection", exc_info=True)


if _pysqlite3 is not None:
    class _PysqliteLockedCursor(_pysqlite3.Cursor):
        """Equivalent cursor guard for the optional pysqlite3 DB-API."""

        def _call(
            self, operation: Callable[[], Any], *, exclusive: bool = False,
        ) -> Any:
            connection = _pysqlite3.Cursor.connection.__get__(self, type(self))
            return connection._sqlite_call(operation, exclusive=exclusive)

        def execute(self, *args: Any, **kwargs: Any):
            self._call(
                lambda: _pysqlite3.Cursor.execute(self, *args, **kwargs)
            )
            return self

        def executemany(self, *args: Any, **kwargs: Any):
            self._call(
                lambda: _pysqlite3.Cursor.executemany(self, *args, **kwargs)
            )
            return self

        def executescript(self, *args: Any, **kwargs: Any):
            self._call(
                lambda: _pysqlite3.Cursor.executescript(self, *args, **kwargs)
            )
            return self

        def fetchone(self) -> Any:
            return self._call(lambda: _pysqlite3.Cursor.fetchone(self))

        def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
            return self._call(
                lambda: _pysqlite3.Cursor.fetchmany(self, *args, **kwargs)
            )

        def fetchall(self) -> Any:
            return self._call(lambda: _pysqlite3.Cursor.fetchall(self))

        def close(self) -> Any:
            return self._call(lambda: _pysqlite3.Cursor.close(self))

        def __iter__(self):
            return self

        def __next__(self) -> Any:
            return self._call(lambda: _pysqlite3.Cursor.__next__(self))

        def __del__(self) -> None:
            try:
                connection = _pysqlite3.Cursor.connection.__get__(
                    self, type(self),
                )
                if getattr(connection, "_sqlite_closed", False):
                    return
                self._call(lambda: _pysqlite3.Cursor.close(self))
            except Exception:
                logger.debug("failed to finalize pysqlite cursor", exc_info=True)


    class _PysqliteLockedConnection(_pysqlite3.Connection):
        """Equivalent connection guard for the optional pysqlite3 DB-API."""

        _sqlite_call = _LockedConnection._sqlite_call

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._sqlite_closed = False
            self._sqlite_gate = _SQLiteCallGate()

        def _sqlite_in_transaction(self) -> bool:
            try:
                return bool(
                    _pysqlite3.Connection.in_transaction.__get__(
                        self, type(self),
                    )
                )
            except Exception:  # pragma: no cover - closed or half-built handle
                return False

        def cursor(self, *args: Any, **kwargs: Any):
            factory = kwargs.pop("factory", _PysqliteLockedCursor)
            return self._sqlite_call(
                lambda: _pysqlite3.Connection.cursor(
                    self, *args, factory=factory, **kwargs,
                )
            )

        def _statement(
            self, method: Callable[..., Any], *args: Any, **kwargs: Any,
        ):
            def run():
                cursor = _pysqlite3.Connection.cursor(
                    self, factory=_PysqliteLockedCursor,
                )
                try:
                    method(cursor, *args, **kwargs)
                except BaseException:
                    _pysqlite3.Cursor.close(cursor)
                    raise
                return cursor

            return self._sqlite_call(run)

        def execute(self, *args: Any, **kwargs: Any):
            return self._statement(
                _pysqlite3.Cursor.execute, *args, **kwargs,
            )

        def executemany(self, *args: Any, **kwargs: Any):
            return self._statement(
                _pysqlite3.Cursor.executemany, *args, **kwargs,
            )

        def executescript(self, *args: Any, **kwargs: Any):
            return self._statement(
                _pysqlite3.Cursor.executescript, *args, **kwargs,
            )

        def commit(self) -> Any:
            return self._sqlite_call(
                lambda: _pysqlite3.Connection.commit(self)
            )

        def rollback(self) -> Any:
            return self._sqlite_call(
                lambda: _pysqlite3.Connection.rollback(self)
            )

        def close(self) -> Any:
            if self._sqlite_closed:
                return None

            def close_once() -> Any:
                if self._sqlite_closed:
                    return None
                result = _pysqlite3.Connection.close(self)
                self._sqlite_closed = True
                return result

            return self._sqlite_call(
                close_once, exclusive=True,
            )

        def create_function(self, *args: Any, **kwargs: Any) -> Any:
            return self._sqlite_call(
                lambda: _pysqlite3.Connection.create_function(
                    self, *args, **kwargs,
                )
            )

        def __enter__(self):
            self._sqlite_call(
                lambda: _pysqlite3.Connection.__enter__(self)
            )
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
            return self._sqlite_call(
                lambda: _pysqlite3.Connection.__exit__(
                    self, exc_type, exc, traceback,
                )
            )

        def interrupt(self) -> Any:
            return _pysqlite3.Connection.interrupt(self)

        def __del__(self) -> None:
            if getattr(self, "_sqlite_closed", False):
                return
            try:
                self.close()
            except Exception:
                logger.debug("failed to finalize pysqlite connection", exc_info=True)
else:  # Keep the name available for factory-selection code and type checks.
    _PysqliteLockedConnection = None


class _SerializedCursor(Iterator[Any]):
    """Fallback cursor proxy for connect wrappers used by instrumentation."""

    def __init__(self, cursor: Any, connection: "_SerializedConnection"):
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "_connection", connection)

    def _call(
        self, operation: Callable[[], Any], *, exclusive: bool = False,
    ) -> Any:
        return self._connection._sqlite_call(operation, exclusive=exclusive)

    def execute(self, *args: Any, **kwargs: Any) -> "_SerializedCursor":
        self._call(lambda: self._cursor.execute(*args, **kwargs))
        return self

    def executemany(self, *args: Any, **kwargs: Any) -> "_SerializedCursor":
        self._call(lambda: self._cursor.executemany(*args, **kwargs))
        return self

    def executescript(self, *args: Any, **kwargs: Any) -> "_SerializedCursor":
        self._call(lambda: self._cursor.executescript(*args, **kwargs))
        return self

    def fetchone(self) -> Any:
        return self._call(self._cursor.fetchone)

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(lambda: self._cursor.fetchmany(*args, **kwargs))

    def fetchall(self) -> Any:
        return self._call(self._cursor.fetchall)

    def close(self) -> Any:
        return self._call(self._cursor.close)

    @property
    def connection(self) -> "_SerializedConnection":
        return self._connection

    def __iter__(self) -> "_SerializedCursor":
        return self

    def __next__(self) -> Any:
        return self._call(lambda: next(self._cursor))

    def __getattr__(self, name: str) -> Any:
        return self._call(lambda: getattr(self._cursor, name))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_cursor", "_connection"}:
            object.__setattr__(self, name, value)
            return
        self._call(lambda: setattr(self._cursor, name, value))

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        cursor = getattr(self, "_cursor", None)
        if connection is None or cursor is None:
            return
        if getattr(connection, "_sqlite_closed", False):
            return
        try:
            connection._sqlite_call(cursor.close)
        except Exception:
            logger.debug("failed to finalize serialized SQLite cursor",
                         exc_info=True)


class _SerializedConnection:
    """Fallback proxy when a test/instrumentation wrapper hides the subclass."""

    def __init__(self, connection: Any):
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "_sqlite_closed", False)
        object.__setattr__(self, "_sqlite_gate", _SQLiteCallGate())

    def _sqlite_in_transaction(self) -> bool:
        try:
            return bool(getattr(self._connection, "in_transaction", False))
        except Exception:  # pragma: no cover - wrapper without the attribute
            return False

    def _sqlite_call(
        self, operation: Callable[[], Any], *, exclusive: bool = False,
    ) -> Any:
        self._sqlite_gate.enter(
            exclusive=exclusive,
            holds_database_lock=not exclusive and self._sqlite_in_transaction(),
        )
        try:
            return operation()
        finally:
            self._sqlite_gate.leave(exclusive=exclusive)

    def _wrap_cursor(self, cursor: Any) -> Any:
        if cursor is None or isinstance(cursor, _SerializedCursor):
            return cursor
        if hasattr(cursor, "fetchone") or isinstance(cursor, sqlite3.Cursor):
            return _SerializedCursor(cursor, self)
        return cursor

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrap_cursor(
            self._sqlite_call(lambda: self._connection.cursor(*args, **kwargs))
        )

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrap_cursor(
            self._sqlite_call(lambda: self._connection.execute(*args, **kwargs))
        )

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrap_cursor(
            self._sqlite_call(lambda: self._connection.executemany(*args, **kwargs))
        )

    def executescript(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrap_cursor(
            self._sqlite_call(lambda: self._connection.executescript(*args, **kwargs))
        )

    def commit(self) -> Any:
        return self._sqlite_call(self._connection.commit)

    def rollback(self) -> Any:
        return self._sqlite_call(self._connection.rollback)

    def close(self) -> Any:
        if self._sqlite_closed:
            return None

        def close_once() -> Any:
            if self._sqlite_closed:
                return None
            result = self._connection.close()
            self._sqlite_closed = True
            return result

        return self._sqlite_call(close_once, exclusive=True)

    def __getattr__(self, name: str) -> Any:
        attribute = self._sqlite_call(lambda: getattr(self._connection, name))
        if not callable(attribute):
            return attribute

        def guarded(*args: Any, **kwargs: Any) -> Any:
            return self._wrap_cursor(
                self._sqlite_call(lambda: attribute(*args, **kwargs))
            )

        return guarded

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_sqlite_") or name == "_connection":
            object.__setattr__(self, name, value)
            return
        self._sqlite_call(lambda: setattr(self._connection, name, value))

    def __enter__(self) -> "_SerializedConnection":
        self._sqlite_call(self._connection.__enter__)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._sqlite_call(
            lambda: self._connection.__exit__(exc_type, exc, traceback)
        )

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None or getattr(self, "_sqlite_closed", False):
            return
        try:
            self.close()
        except Exception:
            logger.debug("failed to finalize serialized SQLite connection",
                         exc_info=True)


def connect_with_lock(
    connect: Callable[..., _ConnectionT],
    /,
    *args: Any,
    **kwargs: Any,
) -> _ConnectionT:
    """Open SQLite using connection classes whose close cannot overlap their own calls."""
    connect_module = str(getattr(connect, "__module__", ""))
    if connect_module.startswith("pysqlite3."):
        selected_factory = _PysqliteLockedConnection
    elif connect_module in {"_sqlite3", "sqlite3", "sqlite3.dbapi2"}:
        selected_factory = _LockedConnection
    else:
        # Test instrumentation and application adapters may wrap either DB-API.
        # Let the wrapper create its native type, then guard the returned object
        # with the fallback proxy instead of passing an incompatible C factory.
        selected_factory = None
    factory = kwargs.get("factory")
    allowed_factories = {
        item for item in (_LockedConnection, _PysqliteLockedConnection)
        if item is not None
    }
    if factory is not None and factory not in allowed_factories:
        raise ValueError("connect_with_lock requires its guarded connection factory")
    if factory is not None or selected_factory is not None:
        kwargs["factory"] = factory or selected_factory
    connection = connect(*args, **kwargs)
    native_types = tuple(allowed_factories)
    if isinstance(connection, native_types + (_SerializedConnection,)):
        return connection
    return _SerializedConnection(connection)
