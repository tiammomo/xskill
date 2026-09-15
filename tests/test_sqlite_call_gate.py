"""Gate contract: finalization must not starve, and must not deadlock itself."""
from __future__ import annotations

import inspect
import sqlite3
import threading
import time

import pytest

from xskill._sqlite_connect import (
    _SQLITE_CALL_GATE,
    _SQLiteCallGate,
    connect_with_lock,
)


def _run_with_deadline(target, seconds: float) -> bool:
    """Run ``target`` in a thread and report whether it finished in time."""
    finished = threading.Event()

    def guarded():
        target()
        finished.set()

    threading.Thread(target=guarded, daemon=True).start()
    return finished.wait(seconds)


def test_finalize_inside_active_call_is_rejected():
    gate = _SQLiteCallGate()
    gate.enter(exclusive=False)
    try:
        with pytest.raises(RuntimeError, match="cannot finalize"):
            gate.enter(exclusive=True)
    finally:
        gate.leave(exclusive=False)


def test_finalizing_thread_may_reenter_shared_side():
    """A cursor or blob finalizer runs inside ``Connection.close()``."""
    gate = _SQLiteCallGate()

    def finalize_with_teardown():
        gate.enter(exclusive=True)
        try:
            gate.enter(exclusive=False)
            gate.leave(exclusive=False)
        finally:
            gate.leave(exclusive=True)

    assert _run_with_deadline(finalize_with_teardown, 5)


def test_waiting_finalizer_holds_back_new_calls():
    gate = _SQLiteCallGate()
    blocking_call_entered = threading.Event()
    release_blocking_call = threading.Event()

    def blocking_call():
        gate.enter(exclusive=False)
        blocking_call_entered.set()
        release_blocking_call.wait(5)
        gate.leave(exclusive=False)

    threading.Thread(target=blocking_call, daemon=True).start()
    assert blocking_call_entered.wait(5)

    finalizer_waiting = threading.Event()
    finalizer_done = threading.Event()

    def finalize():
        finalizer_waiting.set()
        gate.enter(exclusive=True)
        gate.leave(exclusive=True)
        finalizer_done.set()

    threading.Thread(target=finalize, daemon=True).start()
    assert finalizer_waiting.wait(5)
    time.sleep(0.2)

    latecomer_done = threading.Event()

    def latecomer():
        gate.enter(exclusive=False)
        gate.leave(exclusive=False)
        latecomer_done.set()

    threading.Thread(target=latecomer, daemon=True).start()
    assert not latecomer_done.wait(0.5)

    release_blocking_call.set()
    assert finalizer_done.wait(5)
    assert latecomer_done.wait(5)


def test_nested_call_ignores_waiting_finalizer():
    """An in-flight call re-entering the gate would otherwise deadlock."""
    gate = _SQLiteCallGate()
    outer_call_entered = threading.Event()
    finalizer_waiting = threading.Event()
    nesting_done = threading.Event()

    def call_sqlite_twice():
        gate.enter(exclusive=False)
        outer_call_entered.set()
        finalizer_waiting.wait(5)
        time.sleep(0.2)
        gate.enter(exclusive=False)
        gate.leave(exclusive=False)
        nesting_done.set()
        gate.leave(exclusive=False)

    threading.Thread(target=call_sqlite_twice, daemon=True).start()
    assert outer_call_entered.wait(5)

    def finalize():
        finalizer_waiting.set()
        gate.enter(exclusive=True)
        gate.leave(exclusive=True)

    threading.Thread(target=finalize, daemon=True).start()
    assert nesting_done.wait(5)


def test_lock_holding_call_passes_a_waiting_finalizer():
    """Holding these back keeps the lock the finalizer's peers wait on."""
    gate = _SQLiteCallGate()
    gate.enter(exclusive=False)
    finalizer_waiting = threading.Event()

    def finalize():
        finalizer_waiting.set()
        gate.enter(exclusive=True)
        gate.leave(exclusive=True)

    threading.Thread(target=finalize, daemon=True).start()
    assert finalizer_waiting.wait(5)
    time.sleep(0.2)

    def call_while_holding_locks():
        gate.enter(exclusive=False, holds_database_lock=True)
        gate.leave(exclusive=False)

    try:
        assert _run_with_deadline(call_while_holding_locks, 5)
    finally:
        gate.leave(exclusive=False)


def test_open_transaction_outruns_a_waiting_finalizer(tmp_path):
    """A writer blocked behind a finalizer would never release its lock."""
    db_path = tmp_path / "txn.db"
    writing_connection = connect_with_lock(
        sqlite3.connect, str(db_path), check_same_thread=False,
    )
    writing_connection.execute("CREATE TABLE pending(value TEXT)")
    writing_connection.commit()
    writing_connection.execute("INSERT INTO pending VALUES ('first')")
    assert writing_connection.in_transaction

    finalizer_waiting = threading.Event()
    finalizer_done = threading.Event()
    _SQLITE_CALL_GATE.enter(exclusive=False)
    try:
        def finalize():
            finalizer_waiting.set()
            _SQLITE_CALL_GATE.enter(exclusive=True)
            _SQLITE_CALL_GATE.leave(exclusive=True)
            finalizer_done.set()

        threading.Thread(target=finalize, daemon=True).start()
        assert finalizer_waiting.wait(5)
        time.sleep(0.2)
        assert _run_with_deadline(
            lambda: writing_connection.execute(
                "INSERT INTO pending VALUES ('second')"
            ),
            5,
        )
        assert _run_with_deadline(writing_connection.commit, 5)
    finally:
        _SQLITE_CALL_GATE.leave(exclusive=False)
    assert finalizer_done.wait(5)
    writing_connection.close()


def test_close_completes_under_sustained_read_load(tmp_path):
    """Regression: readers used to overtake a waiting close indefinitely."""
    db_path = tmp_path / "hot.db"
    seed_connection = connect_with_lock(
        sqlite3.connect, str(db_path), check_same_thread=False,
    )
    seed_connection.execute("CREATE TABLE rows_under_load(value TEXT)")
    seed_connection.executemany(
        "INSERT INTO rows_under_load VALUES (?)",
        [("payload" * 20,) for _ in range(5000)],
    )
    seed_connection.commit()
    seed_connection.close()

    stop_readers = threading.Event()
    reader_threads = []

    def keep_reading():
        connection = connect_with_lock(
            sqlite3.connect, str(db_path), check_same_thread=False,
        )
        try:
            while not stop_readers.is_set():
                connection.execute(
                    "SELECT count(*) FROM rows_under_load WHERE value LIKE '%load%'"
                ).fetchall()
        finally:
            connection.close()

    for _ in range(4):
        thread = threading.Thread(target=keep_reading, daemon=True)
        thread.start()
        reader_threads.append(thread)

    closing_connection = connect_with_lock(
        sqlite3.connect, str(db_path), check_same_thread=False,
    )
    closing_connection.execute("SELECT 1").fetchall()
    time.sleep(0.5)
    try:
        assert _run_with_deadline(closing_connection.close, 20)
    finally:
        stop_readers.set()
        for thread in reader_threads:
            thread.join(timeout=10)


def test_trajectory_routes_do_not_run_sqlite_on_the_event_loop():
    """Starlette runs sync routes in its thread pool; async ones block it."""
    import xskill.api.app as api_app

    assert not inspect.iscoroutinefunction(api_app.api_list_trajectories)
    assert not inspect.iscoroutinefunction(api_app.api_trajectory_logs)
