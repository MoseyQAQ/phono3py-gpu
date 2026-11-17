"""Single-node GPU workers for independent initial-q scattering tasks.

Workers are fresh Python subprocesses, so ordinary scripts need no multiprocessing
main guard and never fork an initialized CUDA context. A temporary CPU snapshot
preserves the canonical phonon basis; only NumPy results cross process boundaries.
"""

import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import copy
from pathlib import Path

from phono3py_gpu.backend import cupy


def devices():
    """Return logical device IDs within CUDA_VISIBLE_DEVICES."""
    cp = cupy()
    value = os.environ.get("PHONO3PY_GPU_DEVICES")
    if value is None:
        return [cp.cuda.runtime.getDevice()]
    ids = [int(x.strip()) for x in value.split(",")]
    if len(set(ids)) != len(ids) or any(
        d < 0 or d >= cp.cuda.runtime.getDeviceCount() for d in ids
    ):
        raise ValueError("PHONO3PY_GPU_DEVICES must list distinct visible GPU IDs.")
    return ids


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


def iterate(calculator, solver, ids, indices, q_batch, collision, callback=None):
    """Schedule heavier q groups first, store results in original global slots."""
    pp = calculator._pp
    grid_points = calculator._kappa_settings.grid_points
    counts = {}
    for index in indices:
        pp.set_grid_point(int(grid_points[index]))
        counts[index] = len(pp.get_triplets_at_q()[0])
    groups = [indices[i : i + q_batch] for i in range(0, len(indices), q_batch)]
    groups.sort(key=lambda group: sum(counts[i] for i in group), reverse=True)
    tasks, results = queue.Queue(), queue.Queue()
    last = indices[-1]
    for group in groups:
        tasks.put((group, grid_points[group], last in group))
    options = dict(
        sigmas=solver._sigmas,
        sigma_cutoff=solver._sigma_cutoff,
        temperatures=solver._temperatures,
    )
    payload = dict(pp=copy(pp), collision=collision, options=options, lang=pp._lang)
    if collision:
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
            pickle.dump(payload, stream, protocol=5)
        executor = ThreadPoolExecutor(max_workers=len(ids))
        futures = [
            executor.submit(
                _serve, d, snapshot, tasks, results, processes, cancelled, ready
            )
            for d in ids
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
                        calculator._kappa_solver.store(index, answer)
                    else:
                        pending[index] = answer
                    calculator._grid_point_count += 1
                if state is not None:
                    last_state = state
                if calculator._log_level:
                    print(
                        f"GPU {device}: completed {calculator._grid_point_count}/"
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
                pp.set_grid_point(int(grid_points[index]))
                callback(index)


def _worker():
    from phono3py.phonon3.collision_matrix import CollisionMatrix

    from phono3py_gpu.batching import compute_q_batch
    from phono3py_gpu.interaction import InteractionKernel
    from phono3py_gpu.scattering import GPUCollisionSolver, GPUScatteringSolver

    output, commands = sys.stdout.buffer, sys.stdin.buffer
    with redirect_stdout(sys.stderr):
        start = time.perf_counter()
        cp = cupy()
        cp.cuda.Device(int(sys.argv[1])).use()
        with open(sys.argv[2], "rb") as stream:
            payload = pickle.load(stream)
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
            _, gps, last = task
            solver.q_batch_stats = dict(triplet_batches=0, max_q_per_launch=0)
            start = time.perf_counter()
            answers = compute_q_batch(solver, gps, collision=payload["collision"])
            state = (
                (pp.interaction_strength, pp.zero_value_positions)
                if last and payload["collision"]
                else None
            )
            pickle.dump(
                (answers, time.perf_counter() - start, solver.q_batch_stats, state),
                output,
                protocol=5,
            )
            output.flush()


if __name__ == "__main__":
    _worker()
