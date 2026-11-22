"""Single-node GPU workers for independent initial-q scattering tasks.

The primary GPU reuses the parent solver; other GPUs use fresh subprocesses.
A copy-on-write mapped snapshot shares large host arrays while preserving private
mutable state and the canonical phonon basis. Only NumPy results cross processes.
"""

import mmap
import pickle
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import copy
from pathlib import Path

from phono3py.phonon3.triplets import get_nosym_triplets_at_q, get_triplets_at_q

from phono3py_gpu.backend import cupy


def _write_snapshot(payload, stream):
    """Keep large NumPy buffers outside the pickle for shared mapped loading."""
    buffers = []

    def collect(buffer):
        view = buffer.raw()
        if view.nbytes < 1024**2:
            return True
        buffers.append(view)
        return False

    metadata = pickle.dumps(payload, protocol=5, buffer_callback=collect)
    for view in buffers:
        stream.write(struct.pack("<Q", view.nbytes))
        stream.write(b"\0" * (-stream.tell() % 64))
        stream.write(view)
    offset = stream.tell()
    stream.write(metadata)
    stream.write(struct.pack("<Q", offset))


def _load_snapshot(stream):
    """Workers share file pages; writes remain private through copy-on-write."""
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    view = memoryview(mapping)
    (offset,) = struct.unpack_from("<Q", view, len(view) - 8)

    def buffers():
        position = 0
        while position < offset:
            (size,) = struct.unpack_from("<Q", view, position)
            position = (position + 8 + 63) // 64 * 64
            yield view[position : position + size]
            position += size

    stream.seek(offset)
    return pickle.load(stream, buffers=buffers())


def devices():
    """Use every GPU visible to the CUDA runtime."""
    return list(range(cupy().cuda.runtime.getDeviceCount()))


def _exchange(process, value):
    pickle.dump(value, process.stdin, protocol=5)
    process.stdin.flush()
    return pickle.load(process.stdout)


def _serve(device, snapshot, tasks, results, processes, cancelled, ready):
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "phono3py_gpu.multigpu", str(device), str(snapshot)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        processes.append(process)
        if cancelled.is_set():
            raise RuntimeError("GPU worker startup was cancelled.")
        setup = pickle.load(process.stdout)
        results.put(("ready", device, setup))
        ready.wait()
        while not cancelled.is_set():
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                break
            payload = _exchange(process, task)
            results.put(("result", device, (task, payload)))
        pickle.dump(None, process.stdin, protocol=5)
        process.stdin.flush()
        if process.wait() != 0:
            raise RuntimeError(f"GPU {device} worker exited with {process.returncode}.")
    except BaseException as error:
        results.put(("error", device, error))
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdin.close()
            process.stdout.close()


def _compute_task(solver, task, collision):
    from phono3py_gpu.batching import compute_q_batch

    _, gps, last = task
    solver.q_batch_stats = dict(triplet_batches=0, max_q_per_launch=0)
    start = time.perf_counter()
    answers = compute_q_batch(solver, gps, collision=collision, last_batch=last)
    pp = solver._pp
    state = (
        (pp.interaction_strength, pp.zero_value_positions)
        if last and collision and solver.retain_pp
        else None
    )
    return answers, time.perf_counter() - start, solver.q_batch_stats, state


def _serve_primary(device, solver, collision, tasks, results, cancelled, ready):
    """首卡复用已有声子数组，避免再启动进程复制整张声子网格。"""
    try:
        cupy().cuda.Device(device).use()
        start = time.perf_counter()
        solver.engine.fourier
        results.put(
            (
                "ready",
                device,
                dict(
                    setup_seconds=time.perf_counter() - start,
                    triplet_batch=solver.engine.batch_size,
                ),
            )
        )
        ready.wait()
        while not cancelled.is_set():
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                break
            payload = _compute_task(solver, task, collision)
            results.put(("result", device, (task, payload)))
    except BaseException as error:
        results.put(("error", device, error))
        raise


def iterate(calculator, solver, ids, indices, q_batch, collision, callback=None):
    """Schedule heavier q groups first, store results in original global slots."""
    pp = calculator._pp
    grid_points = calculator._kappa_settings.grid_points
    counts = {}
    get_triplets = (
        get_triplets_at_q if pp._is_mesh_symmetry else get_nosym_triplets_at_q
    )
    for index in indices:
        # Scheduling needs counts only; workers prepare and validate q state.
        counts[index] = len(get_triplets(int(grid_points[index]), pp.bz_grid)[0])
    groups = [indices[i : i + q_batch] for i in range(0, len(indices), q_batch)]
    final_group = groups[-1]
    groups.sort(key=lambda group: sum(counts[i] for i in group), reverse=True)
    tasks, results = queue.Queue(), queue.Queue()
    last = indices[-1]
    for group in groups:
        tasks.put((group, grid_points[group], group is final_group))
    options = dict(
        sigmas=solver._sigmas,
        sigma_cutoff=solver._sigma_cutoff,
        temperatures=solver._temperatures,
    )
    payload = dict(pp=copy(pp), collision=collision, options=options, lang=pp._lang)
    if collision:
        options["retain_pp"] = solver.retain_pp
        payload["rot_grid_points"] = (
            None
            if solver._collision._is_reducible_collision_matrix
            else pp.bz_grid.grg2bzg[solver._collision._rot_grid_points]
        )
        payload["is_kappa_star"] = calculator._kappa_settings.is_kappa_star
    else:
        options["is_N_U"] = calculator._is_N_U
        options["lang"] = pp._lang
    # Previous calculator runs can leave a large last-q interaction tensor.
    payload["pp"].set_interaction_strength(None)
    stats = calculator.multi_gpu_stats = dict(devices=ids, per_device={})
    cancelled, processes = threading.Event(), []
    ready = threading.Barrier(len(ids))
    calculator._grid_point_count = 0
    last_state = None
    pending = {}
    with tempfile.TemporaryDirectory(prefix="phono3py-gpu-") as directory:
        snapshot = Path(directory) / "interaction.pickle"
        with snapshot.open("wb") as stream:
            _write_snapshot(payload, stream)
        executor = ThreadPoolExecutor(max_workers=len(ids))
        # 主线程只收集结果；其余 GPU 保持独立进程，避免争用同一个 GIL。
        futures = [
            executor.submit(
                _serve_primary,
                ids[0],
                solver,
                collision,
                tasks,
                results,
                cancelled,
                ready,
            )
        ] + [
            executor.submit(
                _serve, d, snapshot, tasks, results, processes, cancelled, ready
            )
            for d in ids[1:]
        ]
        try:
            completed = 0
            while completed < len(groups):
                kind, device, value = results.get()
                if kind == "error":
                    raise RuntimeError(
                        f"GPU {device} scattering worker failed."
                    ) from value
                if kind == "ready":
                    stats["per_device"][device] = dict(
                        **value, q_points=0, triplets=0, compute_seconds=0.0
                    )
                    continue
                (group, gps, _), (answers, elapsed, batch_stats, state) = value
                record = stats["per_device"][device]
                record["q_points"] += len(group)
                record["triplets"] += sum(counts[i] for i in group)
                record["compute_seconds"] += elapsed
                calculator.q_batch_stats["q_batches"] += 1
                calculator.q_batch_stats["triplet_batches"] += batch_stats[
                    "triplet_batches"
                ]
                calculator.q_batch_stats["max_q_per_launch"] = max(
                    calculator.q_batch_stats["max_q_per_launch"],
                    batch_stats["max_q_per_launch"],
                )
                for index, answer in zip(group, answers, strict=True):
                    if collision:
                        calculator._gamma[:, :, index] = answer.gamma
                        if calculator._grid_point_indices is None or index == last:
                            calculator._kappa_solver.store(index, answer)
                    else:
                        pending[index] = answer
                    calculator._grid_point_count += 1
                if state is not None:
                    last_state = state
                if calculator._log_level:
                    print(
                        f"phono3py-gpu: GPU {device}: completed "
                        f"{calculator._grid_point_count}/"
                        f"{len(indices)} q points.",
                        flush=True,
                    )
                completed += 1
            for future in futures:
                future.result()
        finally:
            cancelled.set()
            ready.abort()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            executor.shutdown(wait=True)
    solver.q_batch_stats = calculator.q_batch_stats
    pp.set_grid_point(int(grid_points[last]))
    if last_state is not None:
        pp.set_interaction_strength(last_state[0], g_zero=last_state[1])
    if not collision:
        calculator._gamma_detail_at_q = None
        for count, index in enumerate(indices):
            answer = pending.pop(index)
            calculator._store_scattering_result(
                index, answer["result"], answer["gamma_N"], answer["gamma_U"]
            )
            solver._gamma_N, solver._gamma_U = answer["gamma_N"], answer["gamma_U"]
            calculator._grid_point_count = count + 1
            if callback is not None:
                # Match the batched single-GPU callback using computed metadata.
                pp._triplets_at_q = answer["triplets"]
                pp._weights_at_q = answer["weights"]
                callback(index)


def _worker():
    from phono3py.phonon3.collision_matrix import CollisionMatrix

    from phono3py_gpu.interaction import InteractionKernel
    from phono3py_gpu.scattering import GPUCollisionSolver, GPUScatteringSolver

    output, commands = sys.stdout.buffer, sys.stdin.buffer
    with redirect_stdout(sys.stderr):
        start = time.perf_counter()
        cp = cupy()
        cp.cuda.Device(int(sys.argv[1])).use()
        with open(sys.argv[2], "rb") as stream:
            payload = _load_snapshot(stream)
        pp = payload["pp"]
        engine = InteractionKernel(
            pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
        )
        if payload["collision"]:
            collision = CollisionMatrix(
                pp,
                rot_grid_points=payload["rot_grid_points"],
                is_kappa_star=payload["is_kappa_star"],
                lang=payload["lang"],
            )
            solver = GPUCollisionSolver(
                pp, collision, engine=engine, **payload["options"]
            )
        else:
            solver = GPUScatteringSolver(pp, engine=engine, **payload["options"])
        engine.fourier  # Prepare the resident FC3 before taking tasks.
        pickle.dump(
            dict(
                setup_seconds=time.perf_counter() - start,
                triplet_batch=engine.batch_size,
            ),
            output,
            protocol=5,
        )
        output.flush()
        while (task := pickle.load(commands)) is not None:
            pickle.dump(
                _compute_task(solver, task, payload["collision"]),
                output,
                protocol=5,
            )
            output.flush()


if __name__ == "__main__":
    _worker()
