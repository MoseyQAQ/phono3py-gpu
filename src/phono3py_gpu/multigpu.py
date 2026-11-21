"""Single-node GPU workers for independent initial-q scattering tasks.

Workers share immutable host force constants, while each owns its q state,
phonons, CUDA stream and device allocations. Only NumPy results leave a worker.
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy

from phono3py_gpu.backend import cupy


def devices():
    """Use every GPU visible to the CUDA runtime."""
    return list(range(cupy().cuda.runtime.getDeviceCount()))


def _copy_interaction(pp):
    """Share immutable FC3 buffers, but copy mutable phonons and per-q state."""
    shared = (pp.fc3, pp.fc3_nonzero_indices, pp.all_shortest)
    return deepcopy(pp, {id(array): array for array in shared})


def _serve(device, payload, tasks, results, cancelled, ready):
    from phono3py.phonon3.collision_matrix import CollisionMatrix

    from phono3py_gpu.batching import compute_q_batch
    from phono3py_gpu.interaction import InteractionKernel
    from phono3py_gpu.scattering import GPUCollisionSolver, GPUScatteringSolver

    try:
        start = time.perf_counter()
        cp = cupy()
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            pool = cp.cuda.MemoryPool()
            with cp.cuda.using_allocator(pool.malloc):
                pp = _copy_interaction(payload["pp"])
                engine = InteractionKernel(
                    pp,
                    cp.asarray(pp.phonons.frequencies),
                    cp.asarray(pp.phonons.eigenvectors),
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
                    solver = GPUScatteringSolver(
                        pp, engine=engine, **payload["options"]
                    )
                engine.fourier  # Prepare the resident FC3 before taking tasks.
                results.put(
                    ("ready", device, dict(
                        setup_seconds=time.perf_counter() - start,
                        triplet_batch=engine.batch_size,
                    ))
                )
                ready.wait()
                while not cancelled.is_set():
                    try:
                        task = tasks.get_nowait()
                    except queue.Empty:
                        break
                    _, gps, last = task
                    solver.q_batch_stats = dict(triplet_batches=0, max_q_per_launch=0)
                    start = time.perf_counter()
                    answers = compute_q_batch(
                        solver, gps, collision=payload["collision"], last_batch=last
                    )
                    state = (
                        (pp.interaction_strength, pp.zero_value_positions)
                        if last and payload["collision"] and solver.retain_pp
                        else None
                    )
                    payload_result = (
                        answers,
                        time.perf_counter() - start,
                        solver.q_batch_stats,
                        state,
                    )
                    results.put(("result", device, (task, payload_result)))
                cp.cuda.get_current_stream().synchronize()
    except BaseException as error:
        results.put(("error", device, error))
        cancelled.set()
        ready.abort()
        raise


def iterate(calculator, solver, ids, indices, q_batch, collision, callback=None):
    """Schedule heavier q groups first, store results in original global slots."""
    pp = calculator._pp
    grid_points = calculator._kappa_settings.grid_points
    counts = {}
    for index in indices:
        pp.set_grid_point(int(grid_points[index]))
        counts[index] = len(pp.get_triplets_at_q()[0])
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
    cancelled = threading.Event()
    ready = threading.Barrier(len(ids))
    calculator._grid_point_count = 0
    last_state = None
    pending = {}
    executor = ThreadPoolExecutor(max_workers=len(ids))
    futures = [
        executor.submit(_serve, d, payload, tasks, results, cancelled, ready)
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
