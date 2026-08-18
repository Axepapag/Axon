"""Explicit resource assembly for the continuously ticking Axon runtime.

Importing this module is side-effect free.  Filesystem directories, locks,
SQLite connections, checkpoint loads, and logical-core state are created only
when :func:`materialize_runtime` is called.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Callable, Mapping

from runtime.field import SharedFieldSnapshot

from .config import ModelPin, RuntimeBootstrapConfig
from .contracts import CoreIdentity, RuntimeHead
from .core_backend import CoreBackend, SharedAxonModel
from .dormant import DormantStore
from .engine import (
    ActiveFieldProjector,
    AxonRuntimeEngine,
    IngressSystemUpdateProvider,
)
from .exact_driver import (
    ExactV4RuntimeDriver,
    initialize_logical_core_state,
)
from .idle import DormantIdleWorker, DormantPostCommitHook
from .ingress import IngressQueue
from .projection import ProjectionPolicy
from .soul_store import PrivateRuntimeStateStore, SoulBlobStore
from .store import RuntimeStore
from .contracts import NotInitializedError


class RuntimeBootstrapError(RuntimeError):
    """Runtime resources could not be assembled without violating evidence."""


class RuntimeAlreadyRunningError(RuntimeBootstrapError):
    """Another process or thread owns the runtime runner lock."""


class RuntimeModelPinError(RuntimeBootstrapError):
    """A loaded checkpoint disagrees with its strict bootstrap pin."""


class RuntimeIdentityMismatchError(RuntimeBootstrapError):
    """Persisted runtime identities differ from the configured identity ring."""


class RuntimeConfigMismatchError(RuntimeBootstrapError):
    """The canonical field was initialized by a different runtime config."""


class UnsupportedRuntimeConfigError(RuntimeBootstrapError):
    """A valid descriptor requests behavior this bootstrap cannot yet honor."""


_IN_PROCESS_LOCKS: set[str] = set()
_IN_PROCESS_LOCKS_GUARD = threading.Lock()


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


class RuntimeRunnerLock:
    """One non-blocking, cross-process lock held for a materialized runtime."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._handle = None
        self._key = _path_key(self.path)

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    @staticmethod
    def _lock(handle: object) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: object) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self, *, config_id: str) -> None:
        if self.acquired:
            raise RuntimeAlreadyRunningError("runner lock is already held")
        with _IN_PROCESS_LOCKS_GUARD:
            if self._key in _IN_PROCESS_LOCKS:
                raise RuntimeAlreadyRunningError(
                    f"runtime runner lock is already held: {self.path}"
                )
            _IN_PROCESS_LOCKS.add(self._key)

        handle = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b", buffering=0)
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                os.fsync(handle.fileno())
            try:
                self._lock(handle)
            except (OSError, BlockingIOError) as exc:
                raise RuntimeAlreadyRunningError(
                    f"another runtime owns runner lock {self.path}"
                ) from exc

            metadata = json.dumps(
                {
                    "schema": "axon-runtime-runner-lock-v1",
                    "config_id": config_id,
                    "pid": os.getpid(),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            handle.seek(0)
            handle.truncate()
            handle.write(metadata)
            handle.flush()
            os.fsync(handle.fileno())
            self._handle = handle
        except Exception:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            with _IN_PROCESS_LOCKS_GUARD:
                _IN_PROCESS_LOCKS.discard(self._key)
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            self._unlock(handle)
        finally:
            try:
                handle.close()
            finally:
                with _IN_PROCESS_LOCKS_GUARD:
                    _IN_PROCESS_LOCKS.discard(self._key)

    def __enter__(self) -> "RuntimeRunnerLock":
        if not self.acquired:
            raise RuntimeBootstrapError(
                "runner lock must be explicitly acquired before entering"
            )
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


ModelLoader = Callable[
    [CoreBackend, ModelPin, str],
    SharedAxonModel,
]


def _load_model(
    backend: CoreBackend,
    pin: ModelPin,
    device: str,
) -> SharedAxonModel:
    return backend.load_model(
        model_id=pin.model_id,
        checkpoint_path=pin.checkpoint_path,
        checkpoint_sha256=pin.checkpoint_sha256,
        device=device,
    )


def _connect_dormant(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path), timeout=30.0)


@dataclass(frozen=True, slots=True)
class BootstrapDependencies:
    """Injectable constructors used by focused tests and offline validation."""

    backend_factory: Callable[[], CoreBackend] = CoreBackend
    model_loader: ModelLoader = _load_model
    runtime_store_factory: Callable[[Path], RuntimeStore] = RuntimeStore
    soul_store_factory: Callable[[Path], SoulBlobStore] = SoulBlobStore
    private_state_store_factory: Callable[
        [Path],
        PrivateRuntimeStateStore,
    ] = PrivateRuntimeStateStore
    dormant_connection_factory: Callable[
        [Path],
        sqlite3.Connection,
    ] = _connect_dormant


@dataclass(slots=True)
class MaterializedRuntime:
    """All live resources for one explicitly started runtime process."""

    config: RuntimeBootstrapConfig
    initial_head: RuntimeHead
    identities: tuple[CoreIdentity, ...]
    backend: CoreBackend
    store: RuntimeStore
    ingress_queue: IngressQueue
    soul_store: SoulBlobStore
    private_state_store: PrivateRuntimeStateStore
    dormant_connection: sqlite3.Connection
    dormant_store: DormantStore
    driver: ExactV4RuntimeDriver
    projector: ActiveFieldProjector
    idle_hook: DormantPostCommitHook
    engine: AxonRuntimeEngine
    runner_lock: RuntimeRunnerLock = field(repr=False)
    initialized_new_store: bool = False
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.dormant_connection.close()
        finally:
            try:
                self.store.close()
            finally:
                self.runner_lock.release()

    def __enter__(self) -> "MaterializedRuntime":
        if self._closed:
            raise RuntimeBootstrapError("materialized runtime is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _verify_loaded_models(
    config: RuntimeBootstrapConfig,
    *,
    backend: CoreBackend,
    loader: ModelLoader,
    device: str,
) -> dict[str, SharedAxonModel]:
    loaded: dict[str, SharedAxonModel] = {}
    for pin in config.models:
        shared = loader(backend, pin, device)
        if not isinstance(shared, SharedAxonModel):
            raise RuntimeModelPinError(
                f"model loader returned an invalid value for {pin.model_id!r}"
            )
        checkpoint = shared.checkpoint
        if shared.model_id != pin.model_id:
            raise RuntimeModelPinError("loaded model ID differs from its pin")
        if backend.shared_model(pin.model_id) is not shared:
            raise RuntimeModelPinError(
                "model loader did not register the returned shared model"
            )
        if checkpoint.checkpoint_sha256 != pin.checkpoint_sha256:
            raise RuntimeModelPinError(
                f"checkpoint hash mismatch for {pin.model_id!r}"
            )
        if _path_key(checkpoint.path) != _path_key(pin.checkpoint_path):
            raise RuntimeModelPinError(
                f"checkpoint path mismatch for {pin.model_id!r}"
            )
        if checkpoint.step != pin.checkpoint_step:
            raise RuntimeModelPinError(
                f"checkpoint step mismatch for {pin.model_id!r}: "
                f"expected {pin.checkpoint_step}, got {checkpoint.step}"
            )
        if checkpoint.cfg.d_model != pin.d_model:
            raise RuntimeModelPinError(
                f"checkpoint d_model mismatch for {pin.model_id!r}: "
                f"expected {pin.d_model}, got {checkpoint.cfg.d_model}"
            )
        loaded[pin.model_id] = shared
    return loaded


def _build_identities(
    config: RuntimeBootstrapConfig,
    loaded: Mapping[str, SharedAxonModel],
) -> tuple[CoreIdentity, ...]:
    descriptors = {item.core_id: item for item in config.cores}
    identities: list[CoreIdentity] = []
    for core_id in config.ring.core_ids:
        descriptor = descriptors[core_id]
        pin = config.model_pin(descriptor.model_id)
        checkpoint = loaded[descriptor.model_id].checkpoint
        identities.append(
            CoreIdentity(
                core_id=descriptor.core_id,
                display_name=descriptor.display_name,
                base_checkpoint_path=str(pin.checkpoint_path),
                base_checkpoint_sha256=pin.checkpoint_sha256,
                model_id=descriptor.model_id,
                core_state_sha256=checkpoint.core_state_sha256,
                lineage=descriptor.lineage,
                soul_id=descriptor.soul_id,
                adapter_namespace=descriptor.adapter_namespace,
                enabled=descriptor.enabled,
                role_capabilities=descriptor.role_capabilities,
            )
        )
    return tuple(identities)


def _initialize_new_store(
    *,
    config: RuntimeBootstrapConfig,
    backend: CoreBackend,
    identities: tuple[CoreIdentity, ...],
    store: RuntimeStore,
    soul_store: SoulBlobStore,
    private_state_store: PrivateRuntimeStateStore,
) -> RuntimeHead:
    genesis = SharedFieldSnapshot.empty(
        tick_id=0,
        source_manifest_ids=(config.config_id,),
    )
    descriptors = {item.core_id: item for item in config.cores}
    initial_states = []
    for identity in identities:
        descriptor = descriptors[identity.core_id]
        logical = backend.create_logical_core(
            model_id=identity.model_id,
            core_id=identity.core_id,
            soul_id=identity.soul_id,
            committed_field_id=genesis.field_id,
            adapter_manifest={
                "schema": "axon-runtime-adapter-manifest-v1",
                "namespace": descriptor.adapter_namespace,
                "adapters": [],
            },
            rng_manifest={
                "schema": "axon-runtime-rng-manifest-v1",
                "stream_id": identity.core_id,
                "generation": 0,
            },
        )
        initial_states.append(
            initialize_logical_core_state(
                logical=logical,
                identity=identity,
                genesis=genesis,
                soul_store=soul_store,
                private_state_store=private_state_store,
            )
        )
    return store.initialize(genesis, identities, tuple(initial_states))


def _require_same_identities(
    expected: tuple[CoreIdentity, ...],
    persisted: tuple[CoreIdentity, ...],
) -> None:
    if persisted == expected:
        return
    expected_ids = tuple(item.core_id for item in expected)
    persisted_ids = tuple(item.core_id for item in persisted)
    raise RuntimeIdentityMismatchError(
        "persisted runtime identity ring does not match bootstrap config: "
        f"expected={expected_ids}, persisted={persisted_ids}"
    )


def _require_same_config(
    config: RuntimeBootstrapConfig,
    head: RuntimeHead,
) -> None:
    if config.config_id in head.snapshot.source_manifest_ids:
        return
    raise RuntimeConfigMismatchError(
        "runtime head is not bound to bootstrap config "
        f"{config.config_id}"
    )


def materialize_runtime(
    config: RuntimeBootstrapConfig,
    *,
    device: str = "cpu",
    dependencies: BootstrapDependencies | None = None,
) -> MaterializedRuntime:
    """Explicitly load, verify, restore, and assemble one runtime process."""

    if not isinstance(config, RuntimeBootstrapConfig):
        raise TypeError("config must be RuntimeBootstrapConfig")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if config.idle.max_quanta_per_tick != 1:
        raise UnsupportedRuntimeConfigError(
            "bootstrap currently supports exactly one idle quantum per tick"
        )
    deps = BootstrapDependencies() if dependencies is None else dependencies
    if not isinstance(deps, BootstrapDependencies):
        raise TypeError("dependencies must be BootstrapDependencies")

    runner_lock = RuntimeRunnerLock(
        config.paths.state_root / "axon-runtime.runner.lock"
    )
    runner_lock.acquire(config_id=config.config_id)
    store: RuntimeStore | None = None
    dormant_connection: sqlite3.Connection | None = None
    try:
        backend = deps.backend_factory()
        if not isinstance(backend, CoreBackend):
            raise RuntimeBootstrapError(
                "backend factory did not return CoreBackend"
            )
        loaded = _verify_loaded_models(
            config,
            backend=backend,
            loader=deps.model_loader,
            device=device,
        )
        identities = _build_identities(config, loaded)

        soul_store = deps.soul_store_factory(config.paths.soul_store)
        private_state_store = deps.private_state_store_factory(
            config.paths.private_state_store
        )
        store = deps.runtime_store_factory(config.paths.runtime_db)
        try:
            head = store.recover()
        except NotInitializedError:
            head = _initialize_new_store(
                config=config,
                backend=backend,
                identities=identities,
                store=store,
                soul_store=soul_store,
                private_state_store=private_state_store,
            )
            initialized_new = True
        else:
            _require_same_identities(
                identities,
                store.persisted_identities(),
            )
            _require_same_config(config, head)
            initialized_new = False

        driver = ExactV4RuntimeDriver(
            backend=backend,
            soul_store=soul_store,
            private_state_store=private_state_store,
            identities=identities,
        )
        # Read every content-addressed private artifact before returning a live
        # engine.  New and recovered stores therefore share the same restart
        # validation boundary.
        driver.reconcile_head(head)

        ingress_queue = IngressQueue(store.connection)
        dormant_connection = deps.dormant_connection_factory(
            config.paths.dormant_db
        )
        if not isinstance(dormant_connection, sqlite3.Connection):
            raise RuntimeBootstrapError(
                "dormant connection factory returned an invalid value"
            )
        dormant_store = DormantStore(dormant_connection)
        policy = ProjectionPolicy(
            global_char_budget=config.projection.global_char_budget,
            max_selected_spans=config.projection.max_selected_spans,
            region_char_budgets=config.projection.region_char_budgets,
            pinned_regions=config.projection.pinned_regions,
        )
        projector = ActiveFieldProjector(
            dormant_store=dormant_store,
            policy=policy,
        )
        idle_hook = DormantPostCommitHook(
            DormantIdleWorker(dormant_store),
            policy=policy,
            capture_limit=config.idle.capture_limit_per_quantum,
            job_limit=config.idle.job_limit_per_quantum,
            lease_ticks=config.idle.lease_ticks,
        )
        engine = AxonRuntimeEngine(
            store=store,
            identities=identities,
            driver=driver,
            system_updates=IngressSystemUpdateProvider(ingress_queue),
            projector=projector,
            post_commit_hook=idle_hook,
        )
        return MaterializedRuntime(
            config=config,
            initial_head=head,
            identities=identities,
            backend=backend,
            store=store,
            ingress_queue=ingress_queue,
            soul_store=soul_store,
            private_state_store=private_state_store,
            dormant_connection=dormant_connection,
            dormant_store=dormant_store,
            driver=driver,
            projector=projector,
            idle_hook=idle_hook,
            engine=engine,
            runner_lock=runner_lock,
            initialized_new_store=initialized_new,
        )
    except Exception:
        if dormant_connection is not None:
            try:
                dormant_connection.close()
            except Exception:
                pass
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        runner_lock.release()
        raise


__all__ = [
    "RuntimeBootstrapError",
    "RuntimeAlreadyRunningError",
    "RuntimeModelPinError",
    "RuntimeIdentityMismatchError",
    "RuntimeConfigMismatchError",
    "UnsupportedRuntimeConfigError",
    "RuntimeRunnerLock",
    "BootstrapDependencies",
    "MaterializedRuntime",
    "materialize_runtime",
]
