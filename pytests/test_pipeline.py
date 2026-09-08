"""
Unit tests for the pipelining primitives in :mod:`nearl.pipeline`.

These tests exercise :class:`PrefetchBuffer` and :class:`AsyncWriter` in
isolation (no CUDA required) so they run on any machine.
"""

import threading
import time

import pytest

from nearl.pipeline import _ERROR, _SENTINEL, AsyncWriter, PrefetchBuffer


# ---------------------------------------------------------------------------
# PrefetchBuffer
# ---------------------------------------------------------------------------
class TestPrefetchBuffer:
    def test_put_get_roundtrip(self):
        buffer = PrefetchBuffer(capacity=2)
        buffer.put(("feature", (1, 2)))
        assert buffer.get() == ("feature", (1, 2))

    def test_get_blocks_until_put(self):
        buffer = PrefetchBuffer(capacity=2)
        received = []

        def consumer():
            received.append(buffer.get())

        thread = threading.Thread(target=consumer)
        thread.start()
        # Give the consumer a moment to block on get()
        time.sleep(0.05)
        assert received == []
        buffer.put("item")
        thread.join(timeout=2)
        assert received == ["item"]

    def test_put_blocks_when_full(self):
        buffer = PrefetchBuffer(capacity=1)
        buffer.put("a")
        put_done = []

        def producer():
            buffer.put("b")
            put_done.append(True)

        thread = threading.Thread(target=producer)
        thread.start()
        time.sleep(0.05)
        # The producer should be blocked because the buffer is full
        assert put_done == []
        assert buffer.get() == "a"
        thread.join(timeout=2)
        assert put_done == [True]
        assert buffer.get() == "b"

    def test_close_wakes_consumer_with_sentinel(self):
        buffer = PrefetchBuffer(capacity=2)
        buffer.put("a")
        buffer.close()
        assert buffer.get() == "a"
        assert buffer.get() is _SENTINEL

    def test_qsize(self):
        buffer = PrefetchBuffer(capacity=4)
        assert len(buffer) == 0
        buffer.put("a")
        buffer.put("b")
        assert buffer.qsize() == 2
        assert len(buffer) == 2


# ---------------------------------------------------------------------------
# AsyncWriter
# ---------------------------------------------------------------------------
class TestAsyncWriter:
    def test_submit_applies_callback(self):
        seen = []

        def callback(*item):
            seen.append(item)

        writer = AsyncWriter(callback, capacity=4)
        writer.submit(1, 2)
        writer.submit(3)
        writer.close()
        assert seen == [(1, 2), (3,)]

    def test_close_flushes_pending_items(self):
        seen = []

        def callback(*item):
            seen.append(item)

        writer = AsyncWriter(callback, capacity=16)
        for i in range(50):
            writer.submit(i)
        writer.close()
        assert seen == [(i,) for i in range(50)]

    def test_callback_error_is_raised_on_close(self):
        def callback(*item):
            raise ValueError("boom")

        writer = AsyncWriter(callback, capacity=4)
        writer.submit(1)
        with pytest.raises(ValueError, match="boom"):
            writer.close()

    def test_writer_runs_in_background_thread(self):
        main_thread_id = threading.get_ident()
        callback_thread_ids = []

        def callback(*item):
            callback_thread_ids.append(threading.get_ident())

        writer = AsyncWriter(callback, capacity=4)
        writer.submit(1)
        writer.close()
        assert callback_thread_ids
        assert callback_thread_ids[0] != main_thread_id


# ---------------------------------------------------------------------------
# Error marker
# ---------------------------------------------------------------------------
def test_error_marker_is_distinct_from_sentinel():
    assert _ERROR is not _SENTINEL
