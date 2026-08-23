"""
tiny-task-runner — Zero-dependency async task pipeline for Python agents.
Retries, timeouts, concurrency control, and dependency graphs — in one file.
"""
import asyncio
import heapq
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

__version__ = "1.0.0"
__all__ = ["TaskRunner", "task", "TaskError", "TaskTimeoutError", "TaskRetryExhaustedError"]


# ─── Exceptions ──────────────────────────────────────────────────────────────


class TaskError(Exception):
    """Base exception for task failures."""

    def __init__(self, task_name: str, original_error: Exception):
        self.task_name = task_name
        self.original_error = original_error
        super().__init__(f"Task '{task_name}' failed: {original_error}")


class TaskTimeoutError(TaskError):
    """Raised when a task exceeds its timeout."""

    def __init__(self, task_name: str, timeout: float):
        self.timeout = timeout
        super().__init__(task_name, TimeoutError(f"exceeded {timeout}s timeout"))


class TaskRetryExhaustedError(TaskError):
    """Raised when all retries for a task are exhausted."""

    def __init__(self, task_name: str, retries: int, last_error: Exception):
        self.retries = retries
        self.last_error = last_error
        super().__init__(task_name, last_error)


# ─── Task Metadata ───────────────────────────────────────────────────────────


@dataclass
class TaskMeta:
    name: str
    priority: int = 0
    retries: int = 0
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    timeout: Optional[float] = None
    depends_on: List[str] = field(default_factory=list)
    fn: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


# ─── Task Decorator ──────────────────────────────────────────────────────────


def task(
    priority: int = 0,
    retries: int = 0,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    timeout: Optional[float] = None,
):
    """Decorator to mark a function as a task with retry/timeout/priority config."""

    def decorator(fn: Callable) -> Callable:
        meta = TaskMeta(
            name=getattr(fn, "__name__", str(uuid.uuid4())),
            priority=priority,
            retries=retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            timeout=timeout,
            fn=fn,
        )
        setattr(fn, "_task_meta", meta)
        return fn

    return decorator


# ─── Priority Queue ───────────────────────────────────────────────────────────


class PriorityQueue:
    """Thread-safe priority queue for tasks."""

    def __init__(self):
        self._heap: List[Tuple[int, int, TaskMeta]] = []
        self._counter = 0
        self._lock = asyncio.Lock()

    async def put(self, meta: TaskMeta) -> None:
        async with self._lock:
            heapq.heappush(self._heap, (meta.priority, -self._counter, meta))
            self._counter += 1

    async def get(self) -> Optional[TaskMeta]:
        async with self._lock:
            if self._heap:
                _, _, meta = heapq.heappop(self._heap)
                return meta
            return None

    async def peek(self) -> Optional[TaskMeta]:
        async with self._lock:
            if self._heap:
                _, _, meta = self._heap[0]
                return meta
            return None

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


# ─── Semaphore-Like Concurrency Control ──────────────────────────────────────


class Semaphore:
    """Async semaphore for concurrency limiting."""

    def __init__(self, limit: int):
        self._limit = limit
        self._current = 0
        self._waiters: List[asyncio.Future] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            if self._current < self._limit:
                self._current += 1
                return
            fut = asyncio.get_event_loop().create_future()
            self._waiters.append(fut)
            await fut
            self._current += 1

    async def release(self) -> None:
        async with self._lock:
            self._current -= 1
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        await self.release()


# ─── Task Runner ─────────────────────────────────────────────────────────────


class TaskRunner:
    """
    Async task runner with concurrency limits, retries, timeouts, and DAG support.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        default_timeout: Optional[float] = None,
        default_retries: int = 0,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ):
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._semaphore = Semaphore(max_concurrency)
        self._pending: PriorityQueue = PriorityQueue()
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._done_events: Dict[str, asyncio.Event] = {}

    # ─── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        tasks: List[Coroutine],
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Run a list of coroutines concurrently up to max_concurrency.
        Returns results in input order. Raises first non-retryable error.
        """
        if not tasks:
            return []

        pending = set()
        results: List[Optional[Any]] = [None] * len(tasks)

        async def run_one(coro: Coroutine, idx: int) -> None:
            async with self._semaphore:
                t = timeout or self.default_timeout
                try:
                    if t:
                        results[idx] = await asyncio.wait_for(coro, timeout=t)
                    else:
                        results[idx] = await coro
                except Exception as e:
                    results[idx] = None
                    raise TaskError(f"task_{idx}", e)

        for idx, t in enumerate(tasks):
            pending.add(asyncio.create_task(run_one(t, idx)))

        done, pending = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)

        # Re-raise first error
        for d in done:
            if d.exception() and not isinstance(d.exception(), asyncio.CancelledError):
                for p in pending:
                    p.cancel()
                raise d.exception()

        return results

    async def run_with_retry(
        self,
        coro: Coroutine,
        retries: int = 0,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        timeout: Optional[float] = None,
    ) -> Any:
        """Run a coroutine with retry + exponential backoff."""
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                t = timeout or self.default_timeout
                if t:
                    return await asyncio.wait_for(coro, timeout=t)
                return await coro
            except Exception as e:
                last_error = e
                if attempt < retries:
                    delay = min(backoff_base * (backoff_base**attempt), backoff_max)
                    await asyncio.sleep(delay)
                else:
                    break
        raise TaskRetryExhaustedError(
            getattr(coro, "__name__", "unknown"), retries, last_error
        )

    async def run_dag(
        self,
        pending: Optional[List[Tuple[str, Callable, List[str]]]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run tasks in topological order respecting dependencies.
        pending: list of (name, callable, [dependency_names])
        Returns {name: result}.
        """
        if pending is None:
            pending = []

        # Build adjacency list and in-degree map
        graph: Dict[str, List[str]] = {name: [] for name, _, _ in pending}
        in_degree: Dict[str, int] = {name: 0 for name, _, _ in pending}
        callables: Dict[str, Callable] = {name: fn for name, fn, _ in pending}

        for name, fn, deps in pending:
            for dep in deps:
                if dep in graph:
                    graph[dep].append(name)
                    in_degree[name] += 1

        results: Dict[str, Any] = {}
        ready: List[str] = [n for n, d in in_degree.items() if d == 0]
        running: Set[asyncio.Task] = set()
        name_to_coro: Dict[asyncio.Task, str] = {}

        async def run_task(name: str) -> None:
            deps_met = all(
                dep not in results
                for dep in [
                    dn for dn, fn, deps in pending if dn == name for dn in deps
                ]
            )

            async with self._semaphore:
                t = timeout or self.default_timeout
                try:
                    coro = callables[name]()
                    if t:
                        results[name] = await asyncio.wait_for(coro, timeout=t)
                    else:
                        results[name] = await coro
                except Exception as e:
                    results[name] = None
                    raise TaskError(name, e)

        while ready or running:
            # Start ready tasks
            for name in ready:
                coro = run_task(name)
                t = asyncio.create_task(coro)
                running.add(t)
                name_to_coro[t] = name
            ready.clear()

            if not running:
                break

            done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

            for d in done:
                name = name_to_coro.pop(d)
                if d.exception() and not isinstance(d.exception(), asyncio.CancelledError):
                    for remaining in running:
                        remaining.cancel()
                    raise d.exception()
                # Find newly ready tasks
                for nxt in graph.get(name, []):
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0:
                        ready.append(nxt)

        return results

    def enqueue(self, fn: Callable, *args, **kwargs) -> None:
        """Add a task to the priority queue."""
        meta = TaskMeta(
            name=getattr(fn, "__name__", str(uuid.uuid4())),
            fn=fn,
            args=args,
            kwargs=kwargs,
        )
        asyncio.create_task(self._pending.put(meta))

    async def run_pending(self, timeout: Optional[float] = None) -> List[Any]:
        """Run all queued tasks in priority order."""
        results = []
        while not self._pending.is_empty():
            meta = await self._pending.get()
            if meta is None:
                break
            coro = meta.fn(*meta.args, **meta.kwargs)
            result = await self.run_with_retry(
                coro,
                retries=meta.retries,
                backoff_base=meta.backoff_base,
                backoff_max=meta.backoff_max,
                timeout=timeout or meta.timeout or self.default_timeout,
            )
            results.append(result)
        return results


# ─── Sync TaskRunner (alias for backwards compat) ────────────────────────────


class SyncTaskRunner(TaskRunner):
    """TaskRunner that works with sync functions via ThreadPoolExecutor."""

    def __init__(self, *args, max_workers: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = __import__("concurrent.futures").ThreadPoolExecutor(
            max_workers=max_workers
        )

    async def run_sync(self, fn: Callable, *args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    async def run(
        self,
        tasks: List[Coroutine],
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Convert sync tasks to coroutines and run."""
        converted = []
        for t in tasks:
            if asyncio.iscoroutine(t):
                converted.append(t)
            else:
                converted.append(self.run_sync(t))
        return await super().run(converted, timeout=timeout)
