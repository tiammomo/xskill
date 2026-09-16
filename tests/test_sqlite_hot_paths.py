"""SQLite 高并发读写回归：WAL 不应在每次连接重复切换。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import sqlite3
import threading
import time

import numpy as np
import pytest


class _TrackedConnection:
    def __init__(self, connection, statements: list[str], lock: threading.Lock):
        self._connection = connection
        self._statements = statements
        self._lock = lock

    def execute(self, sql, *args, **kwargs):
        with self._lock:
            self._statements.append(" ".join(str(sql).lower().split()))
        return self._connection.execute(sql, *args, **kwargs)

    @property
    def row_factory(self):
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._connection.row_factory = value

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _tracking_connect(monkeypatch, sqlite_module):
    original_connect = sqlite_module.connect
    statements: list[str] = []
    lock = threading.Lock()

    def connect(*args, **kwargs):
        return _TrackedConnection(original_connect(*args, **kwargs), statements, lock)

    monkeypatch.setattr(sqlite_module, "connect", connect)
    return statements


def _journal_mode_assignments(statements: list[str]) -> list[str]:
    """只统计修改 journal_mode 的 PRAGMA，不把状态查询算进去。"""
    return [sql for sql in statements if sql == "pragma journal_mode=wal"]


class _ConcurrencyProbe:
    """Record peak concurrency in a small, deterministic timing window."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


class _ExecuteProbeConnection:
    def __init__(self, connection, probe: _ConcurrencyProbe):
        self._connection = connection
        self._probe = probe

    def execute(self, sql, *args, **kwargs):
        self._probe.enter()
        try:
            time.sleep(0.01)
            return self._connection.execute(sql, *args, **kwargs)
        finally:
            self._probe.leave()

    @property
    def row_factory(self):
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._connection.row_factory = value

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _slow_connect_probe(monkeypatch, sqlite_module):
    original_connect = sqlite_module.connect
    connect_probe = _ConcurrencyProbe()
    execute_probe = _ConcurrencyProbe()

    def connect(*args, **kwargs):
        connect_probe.enter()
        try:
            time.sleep(0.01)
            connection = original_connect(*args, **kwargs)
        finally:
            connect_probe.leave()
        return _ExecuteProbeConnection(connection, execute_probe)

    monkeypatch.setattr(sqlite_module, "connect", connect)
    return connect_probe, execute_probe


class _FakeCursorOperation:
    def __init__(self, probe: _ConcurrencyProbe):
        self._probe = probe

    def fetchone(self):
        self._probe.enter()
        try:
            time.sleep(0.03)
            return (1,)
        finally:
            self._probe.leave()

    def close(self):
        return None


class _FakeConnectionOperation:
    def __init__(
        self,
        probe: _ConcurrencyProbe,
    ):
        self._probe = probe
        self.closed = threading.Event()

    def execute(self, *_args, **_kwargs):
        return _FakeCursorOperation(self._probe)

    def close(self):
        self._probe.enter()
        try:
            time.sleep(0.03)
            self.closed.set()
        finally:
            self._probe.leave()


def test_close_cannot_overlap_an_active_call():
    """Connection close is isolated from any other SQLite call."""
    from xskill._sqlite_connect import connect_with_lock

    probe = _ConcurrencyProbe()
    barrier = threading.Barrier(2)
    fetch_raw = _FakeConnectionOperation(probe)
    close_raw = _FakeConnectionOperation(probe)
    fetch_conn = connect_with_lock(lambda **_kwargs: fetch_raw)
    close_conn = connect_with_lock(lambda **_kwargs: close_raw)
    cursor = fetch_conn.execute("SELECT 1")

    def fetch_during_close():
        barrier.wait(timeout=10)
        return cursor.fetchone()

    def close_during_fetch():
        barrier.wait(timeout=10)
        return close_conn.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        fetch_future = executor.submit(fetch_during_close)
        close_future = executor.submit(close_during_fetch)
        assert fetch_future.result(timeout=10) == (1,)
        close_future.result(timeout=10)

    assert close_raw.closed.is_set()
    assert probe.max_active == 1
    # Do not leave proxy finalizers to affect a later concurrency assertion.
    cursor.close()
    fetch_conn.close()


def test_fetch_and_connect_overlap_while_no_close_waits():
    """Ordinary SQL work stays concurrent; only finalization serializes it."""
    from xskill._sqlite_connect import connect_with_lock

    probe = _ConcurrencyProbe()
    barrier = threading.Barrier(2)
    fetch_raw = _FakeConnectionOperation(probe)
    fetch_conn = connect_with_lock(lambda **_kwargs: fetch_raw)
    cursor = fetch_conn.execute("SELECT 1")

    def fetch_during_connect():
        barrier.wait(timeout=10)
        return cursor.fetchone()

    def connect_during_fetch():
        barrier.wait(timeout=10)

        def slow_connect(**_kwargs):
            probe.enter()
            try:
                time.sleep(0.03)
                return _FakeConnectionOperation(probe)
            finally:
                probe.leave()

        return connect_with_lock(slow_connect)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fetch_future = executor.submit(fetch_during_connect)
        connect_future = executor.submit(connect_during_fetch)
        assert fetch_future.result(timeout=10) == (1,)
        connected = connect_future.result(timeout=10)

    assert probe.max_active == 2
    cursor.close()
    fetch_conn.close()
    connected.close()


def test_connection_proxy_gc_closes_under_the_operation_lock():
    """Fallback proxy finalization cannot overlap another SQLite C call."""
    from xskill._sqlite_connect import connect_with_lock

    probe = _ConcurrencyProbe()
    raw = _FakeConnectionOperation(probe)
    holder = [connect_with_lock(lambda **_kwargs: raw)]
    started = threading.Event()

    def slow_connect(**_kwargs):
        probe.enter()
        started.set()
        try:
            time.sleep(0.05)
            return _FakeConnectionOperation(probe)
        finally:
            probe.leave()

    def release_last_reference():
        assert started.wait(timeout=10)
        holder.pop()
        gc.collect()

    with ThreadPoolExecutor(max_workers=2) as executor:
        connect_future = executor.submit(connect_with_lock, slow_connect)
        gc_future = executor.submit(release_last_reference)
        connected = connect_future.result(timeout=10)
        gc_future.result(timeout=10)

    assert raw.closed.wait(timeout=10)
    assert probe.max_active == 1
    connected.close()


def test_native_connection_and_cursor_api_remains_compatible():
    """The production factory preserves common sqlite3 types and semantics."""
    from xskill._sqlite_connect import connect_with_lock

    conn = connect_with_lock(sqlite3.connect, ":memory:")
    assert isinstance(conn, sqlite3.Connection)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE sample (value INTEGER)")

    cursor = conn.cursor()
    assert isinstance(cursor, sqlite3.Cursor)
    assert cursor.connection is conn
    assert cursor.execute("SELECT 1 AS value") is cursor
    assert cursor.fetchone()["value"] == 1
    assert [row["value"] for row in conn.execute("SELECT 2 AS value")] == [2]

    with pytest.raises(RuntimeError):
        with conn as entered:
            assert entered is conn
            conn.execute("INSERT INTO sample VALUES (3)")
            raise RuntimeError("rollback")
    assert conn.execute("SELECT count(*) FROM sample").fetchone()[0] == 0
    conn.close()


def test_optional_pysqlite_connection_uses_matching_cursor_types():
    """The server's newer pysqlite3 build must not mix stdlib C types."""
    pysqlite3 = pytest.importorskip("pysqlite3")
    from xskill._sqlite_connect import connect_with_lock

    conn = connect_with_lock(pysqlite3.connect, ":memory:")
    try:
        assert isinstance(conn, pysqlite3.Connection)
        conn.row_factory = pysqlite3.Row
        conn.executescript(
            "CREATE TABLE sample (value INTEGER);"
            "INSERT INTO sample VALUES (1);"
        )
        rows = [row["value"] for row in conn.execute("SELECT value FROM sample")]
        assert rows == [1]
    finally:
        conn.close()


def test_sqlite_operation_lock_is_reentrant_for_user_callbacks():
    """SQLite callbacks may safely use another guarded connection in-thread."""
    from xskill._sqlite_connect import connect_with_lock

    outer = connect_with_lock(sqlite3.connect, ":memory:")
    nested = connect_with_lock(sqlite3.connect, ":memory:")
    try:
        outer.execute("CREATE TABLE active_transaction (value INTEGER)")
        outer.execute("INSERT INTO active_transaction VALUES (1)")
        outer.create_function(
            "nested_value",
            0,
            lambda: nested.execute("SELECT 7").fetchone()[0],
        )
        assert outer.execute("SELECT nested_value()").fetchone()[0] == 7
        outer.rollback()
    finally:
        outer.close()
        nested.close()


def test_write_transaction_keeps_guard_until_commit(tmp_path):
    """A waiting writer cannot block the first writer from committing."""
    from xskill._sqlite_connect import connect_with_lock

    db_path = tmp_path / "transactions.db"
    first = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    second = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    first.execute("CREATE TABLE writes (value INTEGER)")
    first.execute("INSERT INTO writes VALUES (1)")
    second_started = threading.Event()
    second_finished = threading.Event()

    def write_second():
        second_started.set()
        second.execute("INSERT INTO writes VALUES (2)")
        second.commit()
        second_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write_second)
            assert second_started.wait(timeout=5)
            assert not second_finished.wait(timeout=0.05)
            first.commit()
            future.result(timeout=5)
        assert first.execute(
            "SELECT value FROM writes ORDER BY value"
        ).fetchall() == [(1,), (2,)]
    finally:
        first.close()
        second.close()


def test_write_transaction_can_be_committed_from_another_thread(tmp_path):
    """check_same_thread=False may transfer a live transaction safely."""
    from xskill._sqlite_connect import connect_with_lock

    db_path = tmp_path / "cross-thread-commit.db"
    conn = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    other = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    conn.execute("CREATE TABLE writes (value INTEGER)")
    conn.execute("INSERT INTO writes VALUES (1)")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(conn.commit).result(timeout=5)
        other.execute("INSERT INTO writes VALUES (2)")
        other.commit()
        assert conn.execute(
            "SELECT value FROM writes ORDER BY value"
        ).fetchall() == [(1,), (2,)]
    finally:
        conn.close()
        other.close()


def test_wrong_thread_programming_error_does_not_drop_transaction_guard(tmp_path):
    """An invalid cross-thread call neither hangs nor exposes a live writer."""
    from xskill._sqlite_connect import connect_with_lock

    db_path = tmp_path / "wrong-thread.db"
    conn = connect_with_lock(sqlite3.connect, str(db_path), timeout=1)
    other = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    conn.execute("CREATE TABLE writes (value INTEGER)")
    conn.execute("INSERT INTO writes VALUES (1)")

    def write_other():
        other.execute("INSERT INTO writes VALUES (2)")
        other.commit()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            invalid = executor.submit(conn.execute, "SELECT 1")
            with pytest.raises(sqlite3.ProgrammingError):
                invalid.result(timeout=5)

            waiting_writer = executor.submit(write_other)
            time.sleep(0.05)
            assert not waiting_writer.done()
            conn.rollback()
            waiting_writer.result(timeout=5)
    finally:
        conn.close()
        other.close()


def test_constraint_error_keeps_guard_until_rollback(tmp_path):
    """A failed statement can leave SQLite's transaction active."""
    from xskill._sqlite_connect import connect_with_lock

    db_path = tmp_path / "constraint.db"
    first = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    second = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    first.execute("CREATE TABLE writes (value INTEGER UNIQUE)")
    first.execute("INSERT INTO writes VALUES (1)")
    first.commit()
    with pytest.raises(sqlite3.IntegrityError):
        first.execute("INSERT INTO writes VALUES (1)")
    assert first.in_transaction

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting_writer = executor.submit(
                second.execute, "INSERT INTO writes VALUES (2)"
            )
            time.sleep(0.05)
            assert not waiting_writer.done()
            first.rollback()
            waiting_writer.result(timeout=5)
        second.commit()
    finally:
        first.close()
        second.close()


def test_closed_connection_finalizer_does_not_block_live_rollback(tmp_path):
    """Late GC of an already closed connection must not take the close gate."""
    from xskill._sqlite_connect import connect_with_lock

    stale = connect_with_lock(
        sqlite3.connect,
        str(tmp_path / "stale.db"),
        check_same_thread=False,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        stale_future = executor.submit(stale.execute, "SELECT 1")
        stale_future.result(timeout=5)
    stale.close()
    del stale

    db_path = tmp_path / "active.db"
    first = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    second = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    first.execute("CREATE TABLE writes (value INTEGER)")
    first.execute("INSERT INTO writes VALUES (1)")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting_writer = executor.submit(
                second.execute, "INSERT INTO writes VALUES (2)"
            )
            time.sleep(0.05)
            assert not waiting_writer.done()
            del stale_future
            gc.collect()
            first.rollback()
            cursor = waiting_writer.result(timeout=5)
            cursor.close()
        second.commit()
    finally:
        first.close()
        second.close()


def test_fallback_proxy_balances_nested_transaction_guards(tmp_path):
    """Instrumentation wrappers must release proxy and native guard levels."""
    from xskill._sqlite_connect import connect_with_lock

    statements: list[str] = []
    statements_lock = threading.Lock()

    def wrapped_connect(*args, **kwargs):
        return _TrackedConnection(
            sqlite3.connect(*args, **kwargs), statements, statements_lock,
        )

    db_path = tmp_path / "wrapped.db"
    wrapped = connect_with_lock(
        wrapped_connect, str(db_path), timeout=1, check_same_thread=False,
    )
    other = connect_with_lock(
        sqlite3.connect, str(db_path), timeout=1, check_same_thread=False,
    )
    wrapped.execute("CREATE TABLE writes (value INTEGER)")
    wrapped.execute("INSERT INTO writes VALUES (1)")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(wrapped.commit).result(timeout=5)
        other.execute("INSERT INTO writes VALUES (2)")
        other.commit()
        assert wrapped.execute("SELECT count(*) FROM writes").fetchone()[0] == 2
    finally:
        wrapped.close()
        other.close()


def test_cross_thread_gc_of_active_connection_releases_guard(tmp_path):
    """Finalizing an unreachable connection cannot retain the global guard."""
    from xskill._sqlite_connect import connect_with_lock

    db_path = tmp_path / "gc-active.db"
    holder = [connect_with_lock(sqlite3.connect, str(db_path), timeout=1)]
    holder[0].execute("CREATE TABLE writes (value INTEGER)")
    cursor = holder[0].execute("INSERT INTO writes VALUES (1)")
    cursor.close()
    del cursor

    def release_last_reference():
        doomed = holder.pop()
        del doomed
        gc.collect()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(release_last_reference).result(timeout=5)

    conn = connect_with_lock(sqlite3.connect, str(db_path), timeout=1)
    try:
        assert conn.execute("SELECT count(*) FROM writes").fetchone()[0] == 0
    finally:
        conn.close()


def test_connect_open_lock_is_released_when_connect_raises():
    """A failed open must not block the next SQLite connection attempt."""
    from xskill._sqlite_connect import connect_with_lock

    expected = RuntimeError("mock connect failure")

    def fail_connect(**_kwargs):
        raise expected

    with pytest.raises(RuntimeError) as exc_info:
        connect_with_lock(fail_connect)
    assert exc_info.value is expected

    conn = connect_with_lock(sqlite3.connect, ":memory:")
    try:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()


def test_profile_sqlite_opens_and_sql_remain_concurrent(
    tmp_path, monkeypatch,
):
    """30 profile workers can open connections and run SQL concurrently."""
    from xskill.recommend import _sqlite_base
    from xskill.recommend.profile_store import ProfileStore

    store = ProfileStore(tmp_path / "profiles.db")
    connect_probe, execute_probe = _slow_connect_probe(
        monkeypatch, _sqlite_base.sqlite3,
    )
    barrier = threading.Barrier(30)

    def load_profile(index: int) -> None:
        barrier.wait(timeout=10)
        assert store.load(f"client-{index:03d}") is None

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(load_profile, range(30)))

    assert connect_probe.max_active > 1
    assert execute_probe.max_active > 1


def test_registry_sqlite_opens_and_sql_remain_concurrent(
    tmp_path, monkeypatch,
):
    """Registry opens and SQL share the non-finalizing side of the gate."""
    from xskill.pipeline import registry

    db_path = tmp_path / "registry.db"
    registry.get_connection(db_path).close()
    connect_probe, execute_probe = _slow_connect_probe(monkeypatch, registry.sqlite3)
    barrier = threading.Barrier(30)

    def open_registry(_index: int) -> None:
        barrier.wait(timeout=10)
        conn = registry.get_connection(db_path)
        try:
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(open_registry, range(30)))

    assert connect_probe.max_active > 1
    assert execute_probe.max_active > 1


def test_server_sqlite_call_sites_keep_open_and_sql_calls_concurrent(
    tmp_path, monkeypatch,
):
    """Profile, registry, client auth and dashboard opens can overlap."""
    from xskill.dashboard.explore import users_status
    from xskill.pipeline import registry
    from xskill.recommend import _sqlite_base
    from xskill.recommend.profile_store import ProfileStore
    from xskill.team.server.client_registry import ClientRegistry

    profile_store = ProfileStore(tmp_path / "profiles.db")
    registry_path = tmp_path / "registry.db"
    registry.get_connection(registry_path).close()
    client_registry = ClientRegistry(tmp_path / "team_clients.db")
    # All call sites import the same sqlite3 module, so one probe observes
    # every database-open path in this server workload.
    connect_probe, execute_probe = _slow_connect_probe(
        monkeypatch, _sqlite_base.sqlite3,
    )
    barrier = threading.Barrier(40)

    def open_from_server_modules(index: int) -> None:
        barrier.wait(timeout=10)
        if index % 4 == 0:
            assert profile_store.load(f"client-{index:03d}") is None
            return
        if index % 4 == 1:
            conn = registry.get_connection(registry_path)
        elif index % 4 == 2:
            conn = client_registry._conn()
        else:
            assert users_status(registry_path)["users"] == []
            return
        try:
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        finally:
            conn.close()

    try:
        with ThreadPoolExecutor(max_workers=40) as executor:
            list(executor.map(open_from_server_modules, range(40)))
    finally:
        client_registry.close()

    assert connect_probe.max_active > 1
    assert execute_probe.max_active > 1


def test_registry_new_db_concurrent_open_assigns_wal_once(tmp_path, monkeypatch):
    """新 DB 的首批连接并发打开时，WAL 赋值与 schema 初始化不竞态。"""
    from xskill.pipeline import registry

    db_path = tmp_path / "new-registry.db"
    statements = _tracking_connect(monkeypatch, registry.sqlite3)
    barrier = threading.Barrier(30)

    def open_registry(_index: int) -> None:
        barrier.wait(timeout=10)
        conn = registry.get_connection(db_path)
        try:
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trajectories'"
            ).fetchone() is not None
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(open_registry, range(30)))

    assert len(_journal_mode_assignments(statements)) == 1


@pytest.mark.nightly
@pytest.mark.flaky(reruns=5, reruns_delay=2, only_rerun=["database is locked"])
def test_registry_concurrent_usage_writes_do_not_reassign_wal(tmp_path, monkeypatch):
    """600 次 LLM usage 写入期间，热连接不重复赋值 WAL。

    Windows CI 偶发 ``sqlite3.OperationalError: database is locked``（WAL
    并发平台 flake，负载敏感）；仅对该错误重跑，其它失败仍一次挂掉。
    负载敏感故移 nightly（跑通一次即可），不再挡 PR 矩阵。
    """
    from xskill.pipeline import registry

    db_path = tmp_path / "registry.db"
    registry.get_connection(db_path).close()
    statements = _tracking_connect(monkeypatch, registry.sqlite3)

    def write_usage(index: int) -> None:
        registry.record_usage(
            step="skill_edit",
            model="mock-tool-call",
            prompt=index,
            completion=1,
            total=index + 1,
            cost_usd=0,
            price_source="config",
            db_path=db_path,
        )

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(write_usage, range(600)))

    summary = registry.usage_summary(db_path)
    assert summary["total_calls"] == 600
    assert _journal_mode_assignments(statements) == []


def test_profile_and_reco_stores_share_first_use_locks(tmp_path, monkeypatch):
    """多个 store 实例共用新 DB 时，WAL 只赋值一次且 schema 都完整。"""
    from xskill.recommend import _sqlite_base
    from xskill.recommend.profile_store import ProfileStore
    from xskill.recommend.reco_store import RecoStore

    db_path = tmp_path / "new-profile.db"
    statements = _tracking_connect(monkeypatch, _sqlite_base.sqlite3)
    barrier = threading.Barrier(30)

    def create_store(index: int):
        barrier.wait(timeout=10)
        store_type = ProfileStore if index % 2 == 0 else RecoStore
        return store_type(db_path)

    with ThreadPoolExecutor(max_workers=30) as executor:
        stores = list(executor.map(create_store, range(30)))

    assert len(stores) == 30
    assert len(_journal_mode_assignments(statements)) == 1
    conn = _sqlite_base.sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert {"client_interest", "recommendations"} <= tables


def test_profile_store_30_workers_do_not_touch_journal_mode_on_hot_path(
    tmp_path, monkeypatch,
):
    """画像 worker 共享的 store 初始化后，读写不再访问 journal_mode。"""
    from xskill.recommend import _sqlite_base
    from xskill.recommend.profile_store import ProfileStore

    store = ProfileStore(tmp_path / "profiles.db")
    statements = _tracking_connect(monkeypatch, _sqlite_base.sqlite3)

    def refresh_profile(index: int) -> None:
        user_id = f"client-{index:03d}"
        points = np.asarray([[float(index), 1.0]])
        assert store.get_revision(user_id) is None
        assert store.load_vector_cache_entries(user_id, "mock-embed") == {}
        store.upsert(
            user_id,
            feature_tensor=points,
            mean_tensor=points[0],
            used_skills=[],
            points=points,
            point_meta=[{"atom_id": f"atom-{index:03d}", "summary": "mock"}],
            embed_model="mock-embed",
            source_revision=f"rev-{index:03d}",
        )

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(refresh_profile, range(300)))

    assert len(store.all_means()) == 300
    assert not any(sql.startswith("pragma journal_mode") for sql in statements)
