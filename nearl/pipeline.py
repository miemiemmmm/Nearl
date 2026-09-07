"""
Pipelining primitives for overlapping CPU preprocessing with GPU compute.

The :class:`Featurizer` pipeline has three distinct phases per task:

1. **CPU preprocessing** -- trajectory loading, focal-point parsing, weight
   caching and coordinate cropping (``Feature.query``).
2. **GPU compute** -- the CUDA kernels (``Feature.run``).
3. **CPU post-processing** -- writing the results to HDF5 (``Feature.dump``).

These phases are naturally serialized: the GPU sits idle while the CPU queries
and dumps, and the CPU sits idle while the GPU runs. The two primitives in this
module let us overlap them with background threads:

* :class:`PrefetchBuffer` -- a bounded, thread-safe buffer (a small, dependency
  free analogue of a ``torch.utils.data.DataLoader`` prefetch buffer). A
  background *producer* thread runs the CPU preprocessing and deposits the
  resulting GPU tasks with :meth:`PrefetchBuffer.put`; the main process
  *consumes* them with :meth:`PrefetchBuffer.get` and launches the GPU kernels.
  The bounded capacity provides natural back-pressure so the producer never
  runs arbitrarily far ahead of the consumer.

* :class:`AsyncWriter` -- a background thread that drains a queue and applies a
  callback (e.g. ``Feature.dump``) to each item. The main process hands results
  to it with :meth:`AsyncWriter.submit` and continues immediately, so HDF5
  writes overlap with the next GPU kernel launch.

Both primitives are deliberately small, dependency-free and unit-testable.
"""

import queue
import threading

__all__ = ["PrefetchBuffer", "AsyncWriter"]

#: Sentinel used to signal end-of-stream on a :class:`PrefetchBuffer` or
#: :class:`AsyncWriter` queue.
_SENTINEL = object()

#: Marker used to propagate a producer exception to the consumer. The consumer
#: receives ``(_ERROR, exc)`` and re-raises ``exc``.
_ERROR = object()


class PrefetchBuffer:
    """
    A bounded, thread-safe buffer for pipelining CPU preprocessing with GPU
    compute.

    Background producer threads run CPU-bound work and deposit the results with
    :meth:`put`; the main process consumes them with :meth:`get` and runs the
    GPU kernels. The bounded capacity provides back-pressure so producers never
    run arbitrarily far ahead of the consumer.

    Parameters
    ----------
    capacity : int, optional
      The maximum number of items that may be buffered before :meth:`put`
      blocks. Defaults to ``2``.

    Examples
    --------
    >>> buffer = PrefetchBuffer(capacity=2)
    >>> buffer.put(("feature_a", (coords, weights)))
    >>> feature, queried = buffer.get()
    >>> buffer.close()
    """

    def __init__(self, capacity=2):
        self._queue = queue.Queue(maxsize=capacity)

    def put(self, item):
        """
        Deposit an item onto the buffer, blocking if the buffer is full.

        Parameters
        ----------
        item : object
          The item to buffer (typically a ``(feature, queried)`` GPU task).
        """
        self._queue.put(item)

    def get(self):
        """
        Remove and return the next item from the buffer, blocking until one is
        available.

        Returns
        -------
        object
          The next buffered item, or the end-of-stream sentinel once
          :meth:`close` has been called and the buffer is drained.
        """
        return self._queue.get()

    def close(self):
        """
        Signal that no more items will be produced.

        A single end-of-stream sentinel is enqueued so a consumer blocked in
        :meth:`get` wakes up and can terminate its loop.
        """
        self._queue.put(_SENTINEL)

    def qsize(self):
        """
        Return the approximate number of items currently buffered.
        """
        return self._queue.qsize()

    def __len__(self):
        return self._queue.qsize()


class AsyncWriter:
    """
    A background thread that applies a callback to submitted items asynchronously.

    The main process submits results with :meth:`submit` and continues
    immediately; a daemon thread drains the queue and calls ``callback(*item)``
    for each one. This lets HDF5 writes overlap with the next GPU kernel launch.

    Parameters
    ----------
    callback : callable
      The callable invoked as ``callback(*item)`` for each submitted item.
    capacity : int, optional
      The maximum number of items that may be queued before :meth:`submit`
      blocks. Defaults to ``16``.

    Examples
    --------
    >>> writer = AsyncWriter(feature.dump)
    >>> writer.submit(result)
    >>> writer.close()  # flush and stop
    """

    def __init__(self, callback, capacity=16):
        self._callback = callback
        self._queue = queue.Queue(maxsize=capacity)
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="nearl-async-writer", daemon=True
        )
        self._thread.start()

    def submit(self, *item):
        """
        Queue an item for asynchronous processing, blocking if the queue is full.

        Parameters
        ----------
        *item
          The positional arguments passed to the callback.
        """
        self._queue.put(item)

    def _run(self):
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            try:
                self._callback(*item)
            except Exception as exc:  # pragma: no cover - surfaced on close()
                self._error = exc

    def close(self):
        """
        Flush any pending items and stop the background thread.

        Raises
        ------
        Exception
          Re-raises the first exception raised by the callback, if any.
        """
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._error is not None:
            raise self._error
