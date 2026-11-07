# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 4.13b: pre-fixed independent near-contact set for the radar baseline.

This stage deliberately **does not evaluate C or D**. It only proposes full-FK joint
configurations with MuJoCo's own distance and labels them with the independent mesh reference.
The rule (seed, sample counts, bins, path budget) is written first to
``outputs/stage413/b/radar_near_contact_rule.json``. C/D are evaluated later, against the frozen
file this script writes.

Full FK only: every ``q`` is inside the MuJoCo joint limits. No independent rigid-body transform
is mixed in (B class is not used this round). A hull distance is a proposal screen, never a label.
The reference cap 0.03 m is a certified lower bound; a culled sample is classified safe for the
20 mm threshold but is not used as an exact distance.

Run:
    python scripts/ranger_cr10/radar_near_contact_search.py
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.cluster.vq import kmeans2

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collision_parts import constituents_of  # noqa: E402
from reference_geometry import MeshRef  # noqa: E402

ROOT = HERE.parents[1]
SCENE = ROOT / "mujoco/robots/ranger_cr10/scene.xml"
MESH_DIR = ROOT / "rangerboxcr10lidar_description/meshes"
POSES = ROOT / "outputs/stage411/operating_poses.npz"
RULE_PATH = ROOT / "outputs/stage413/b/radar_near_contact_rule.json"
SAFETY = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/safety.py"
ARMS = ("cr10_Link1", "cr10_Link2", "cr10_Link3", "cr10_Link4", "cr10_Link5", "cr10_Link6")
LIDAR = "lidar_link0"
CONTACT_TOL = 1e-4

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="outputs/stage413/b/radar_near_contact")
ap.add_argument("--rule", default=str(RULE_PATH))
ap.add_argument("--workers", type=int, default=10)
args = ap.parse_args()
RULE_PATH = Path(args.rule)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_safety():
    spec = importlib.util.spec_from_file_location("safety_near", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safety = load_safety()
RULE = json.loads(RULE_PATH.read_text())
MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
DATA = mujoco.MjData(MODEL)
BID = {mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(MODEL.nbody)}
ARM_ADR = [MODEL.jnt_qposadr[mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}")] for i in range(1, 7)]
JOINT_IDS = [mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, f"cr10_joint{i}") for i in range(1, 7)]
LO = np.array([MODEL.jnt_range[j][0] for j in JOINT_IDS])
HI = np.array([MODEL.jnt_range[j][1] for j in JOINT_IDS])
POSE_ARRAY = np.load(POSES)["poses"]


def _load_mesh(part):
    if part.kind == "mesh":
        mesh = trimesh.load(str(MESH_DIR / part.stl), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * part.scale
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        from collision_parts import box_mesh

        vertices, faces = box_mesh(part.centre, part.half, part.quat)
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
    return vertices, faces


ARM_REF: dict[str, list[tuple[str, MeshRef]]] = {}
for arm in ARMS:
    ARM_REF[arm] = []
    for part in constituents_of(arm):
        vertices, faces = _load_mesh(part)
        ARM_REF[arm].append((part.name, MeshRef(vertices, faces, name=f"{arm}/{part.name}")))
_vertices, _faces = _load_mesh(constituents_of(LIDAR)[0])
LIDAR_REF = MeshRef(_vertices, _faces, name=LIDAR)


def geom_ids(body: str):
    return [g for g in range(MODEL.ngeom) if MODEL.geom_bodyid[g] == BID[body] and MODEL.geom_contype[g] != 0]


GEOMS = {arm: geom_ids(arm) for arm in ARMS}
LIDAR_GEOMS = geom_ids(LIDAR)


def set_q(q) -> None:
    mujoco.mj_resetDataKeyframe(MODEL, DATA, MODEL.key("home").id)
    for address, value in zip(ARM_ADR, q):
        DATA.qpos[address] = float(value)
    mujoco.mj_forward(MODEL, DATA)


def body_transform(name: str):
    body = BID[name]
    return np.array(DATA.xmat[body], dtype=np.float64).reshape(3, 3), np.array(DATA.xpos[body], dtype=np.float64)


def screen_distance(arm: str) -> float:
    """MuJoCo's own distance for the group; min over atomic collision geoms for Link6."""
    values = [mujoco.mj_geomDistance(MODEL, DATA, g, LIDAR_GEOMS[0], 0.05, None) for g in GEOMS[arm]]
    return float(min(values))


def reference_group(arm: str) -> dict:
    """Independent reference for one arm group at the current q."""
    r_arm, p_arm = body_transform(arm)
    r_lidar, p_lidar = body_transform(LIDAR)
    rotation = r_arm.T @ r_lidar
    translation = r_arm.T @ (p_lidar - p_arm)
    exacts = []
    lowers = []
    failures = []
    contact = False
    contact_name = ""
    details = []
    for name, ref in ARM_REF[arm]:
        result = ref.closest_to_in_frame(LIDAR_REF, rotation, translation, max_clearance=RULE["reference"]["cap_m"])
        if result.intersects:
            contact = True
            contact_name = f"{name}|lidar"
            details.append(f"{name}:intersecting")
        elif result.status == "exact":
            exacts.append((float(result.clearance), f"{name}|lidar"))
            details.append(f"{name}:exact:{float(result.clearance):.9f}")
        elif result.status == "certified_culled":
            lower = float(result.extra.get("lower_bound", result.clearance))
            lowers.append((lower, f"{name}|lidar"))
            details.append(f"{name}:certified_culled:{lower:.9f}")
        else:
            failures.append((result.status, f"{name}|lidar"))
            details.append(f"{name}:{result.status}")
    detail = ";".join(details)
    if contact:
        return dict(status="exact", contact=True, distance=0.0, lower_bound=None, nearest=contact_name,
                    exact_count=0, culled_count=len(lowers), failed_count=len(failures), detail=detail)
    if exacts and not failures:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="exact", contact=False, distance=float(distance), lower_bound=None, nearest=name,
                    exact_count=len(exacts), culled_count=len(lowers), failed_count=0, detail=detail)
    if exacts and failures:
        distance, name = min(exacts, key=lambda item: item[0])
        return dict(status="failed", contact=False, distance=None, lower_bound=None, upper_bound=float(distance),
                    nearest=name, exact_count=len(exacts), culled_count=len(lowers), failed_count=len(failures),
                    detail=detail)
    if lowers and not failures:
        lower, name = min(lowers, key=lambda item: item[0])
        return dict(status="certified_culled", contact=False, distance=None, lower_bound=float(lower),
                    nearest=name, exact_count=0, culled_count=len(lowers), failed_count=0, detail=detail)
    return dict(status="failed", contact=False, distance=None, lower_bound=None, nearest="",
                exact_count=len(exacts), culled_count=len(lowers), failed_count=max(1, len(failures)), detail=detail)


def reference_task(task):
    task_id, arm, q, meta = task
    set_q(q)
    result = reference_group(arm)
    clean_meta = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in meta.items()}
    return dict(task_id=task_id, arm=arm, q=[float(x) for x in q], meta=clean_meta, reference=result)


def bin_of(result: dict) -> str:
    if result["contact"]:
        return "overlap"
    if result["status"] == "exact":
        d = result["distance"]
        for label, (lo, hi) in RULE["bins_m"].items():
            if lo <= d < hi:
                return label
        if d >= RULE["bins_m"]["20-30mm"][1]:
            return ">=30mm_exact"
        return "unknown"
    if result["status"] == "certified_culled":
        return ">=30mm_certified_culled"
    return "failed"


def used_pose_qs():
    used = [1534, 1539, 1583, 1562, 1533]
    extra = json.loads((ROOT / "outputs/stage413/b/radar_extra_poses.json").read_text())
    for item in extra["extra"]:
        if item["pose"] not in used:
            used.append(int(item["pose"]))
    arrays = [POSE_ARRAY[used]]
    near_path = RULE.get("exclusions", {}).get("near_set_json")
    if near_path:
        near = json.loads((ROOT / near_path).read_text())
        arrays.append(np.asarray([row["q"] for row in near["rows"]], dtype=np.float64))
    return np.vstack(arrays)


def excluded(q, used):
    return bool(np.min(np.max(np.abs(used - q), axis=1)) < RULE["exclusions"]["min_linf_rad"])


def a_link1_distance() -> float:
    """A baseline Link1-vs-lidar box distance, only to choose far regression poses."""
    names = list(dict.fromkeys(safety.ARM_BODY_LIST + safety.PROT_BODY_LIST))
    import torch

    position = torch.tensor(np.stack([DATA.xpos[BID[name]] for name in names])[None], dtype=torch.float32)
    quat = torch.tensor(np.stack([DATA.xquat[BID[name]] for name in names])[None], dtype=torch.float32)

    class Asset:
        body_names = names

        class data:  # noqa: N801
            body_pos_w = position
            body_quat_w = quat

    distances = safety.clearance(Asset, cap=0.15)[0].detach().cpu().numpy()
    columns = []
    for k, (i, p) in enumerate(safety.SENSOR_PAIRS):
        if safety.ARM_BODY_LIST[i] == "cr10_Link1" and safety.PROT_GROUPS[p] == LIDAR:
            columns.append(len(safety.SELF_PAIRS) + k)
    return float(min(distances[columns]))


def main() -> None:
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    used = used_pose_qs()
    rng = np.random.default_rng(RULE["seed"])

    # 1. Fixed random pool and MuJoCo proposal screen.
    q_random = rng.uniform(LO, HI, size=(int(RULE["random_joint_samples"]), 6))
    screens = {arm: np.full(len(q_random), np.inf) for arm in ARMS}
    for i, q in enumerate(q_random):
        set_q(q)
        for arm in ARMS:
            screens[arm][i] = screen_distance(arm)
    print("random screen done", flush=True)

    # Joint clusters are a fixed stratification, not a result-dependent choice.
    clusters = min(int(RULE["direct_selection"]["joint_clusters"]), len(q_random))
    centroids, labels = kmeans2(q_random, clusters, iter=50, minit="++", seed=RULE["seed"])
    print("clusters", np.bincount(labels).tolist(), flush=True)

    candidates = []
    for arm in ARMS:
        if arm == "cr10_Link1":
            continue
        for band_index, (lo, hi) in enumerate(RULE["screen"]["bands_m"]):
            for cluster in range(clusters):
                mask = (labels == cluster) & (screens[arm] >= lo) & (screens[arm] < hi)
                if hi >= RULE["screen"]["cap_m"]:
                    mask &= screens[arm] < RULE["screen"]["cap_m"] - 1e-9
                indices = np.flatnonzero(mask)
                if len(indices) == 0:
                    continue
                order = indices[np.argsort(screens[arm][indices])]
                take = int(RULE["direct_selection"]["per_band_per_cluster"][arm])
                for index in order[:take]:
                    q = q_random[index]
                    if excluded(q, used):
                        continue
                    if any(np.max(np.abs(other["q"] - q)) < RULE["exclusions"]["min_linf_rad"]
                           for other in candidates if other["arm"] == arm and other["sample_type"] == "direct_random"):
                        continue
                    candidates.append(dict(arm=arm, q=q.copy(), sample_type="direct_random",
                                           screen_mujoco_m=float(screens[arm][index]),
                                           screen_band=f"{lo}:{hi}", cluster=int(cluster),
                                           source="random_joint_limits"))
    print("direct candidates", len(candidates), flush=True)

    # Link1 has no MuJoCo near candidate (min ~90 mm); use one far operating pose per segment.
    a_rows = []
    for pose in range(len(POSE_ARRAY)):
        set_q(POSE_ARRAY[pose])
        a_rows.append(a_link1_distance())
    a_rows = np.asarray(a_rows)
    for segment in range(6):
        chunk = slice(segment * 300, (segment + 1) * 300)
        order = np.argsort(a_rows[chunk]) + segment * 300
        for pose in order:
            if not excluded(POSE_ARRAY[pose], used):
                candidates.append(dict(arm="cr10_Link1", q=POSE_ARRAY[pose].copy(), sample_type="link1_operating_far",
                                       screen_mujoco_m=None, screen_band="operating_far", cluster=segment,
                                       source=f"operating_pose:{int(pose)}"))
                break

    # 2. Reference labels for the direct candidates; C/D are not computed here.
    max_reference = int(RULE["budget"]["max_reference_queries"])
    if len(candidates) > max_reference:
        print(f"truncating direct candidates {len(candidates)} -> {max_reference}", flush=True)
        candidates = candidates[:max_reference]
    tasks = [(i, item["arm"], item["q"], item) for i, item in enumerate(candidates)]
    reference_results = []
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(reference_task, tasks):
            reference_results.append(result)
    reference_results.sort(key=lambda item: item["task_id"])
    all_reference = dict(
        rule_sha256=sha256(RULE_PATH),
        script_sha256=sha256(Path(__file__)),
        reference_cap_m=RULE["reference"]["cap_m"],
        contact_tolerance_m=CONTACT_TOL,
        task_count=len(tasks),
        rows=[dict(task_id=item["task_id"], arm=item["arm"], q=item["q"], meta=item["meta"],
                   reference=item["reference"], bin=bin_of(item["reference"])) for item in reference_results],
    )
    (out_dir / "near_reference_candidates.json").write_text(json.dumps(all_reference, indent=1))

    # 3. Freeze the direct set (distinct, reference-labelled).
    direct_rows = []
    for item in reference_results:
        meta = item["meta"]
        result = item["reference"]
        direct_rows.append(dict(
            row_id=f"D{len(direct_rows):04d}", pair=f"{item['arm']} {LIDAR}", arm=item["arm"],
            sample_type=meta["sample_type"], source=meta["source"], q=item["q"],
            cluster=meta["cluster"], screen_mujoco_m=meta["screen_mujoco_m"], screen_band=meta["screen_band"],
            reference_status=result["status"], reference_contact=bool(result["contact"]),
            reference_distance_m=result["distance"], reference_lower_bound_m=result["lower_bound"],
            reference_threshold_m=RULE["reference"]["cap_m"], reference_nearest=result["nearest"],
            reference_detail=result["detail"], bin=bin_of(result),
            path_id=None, direction=None, s=None, s_boundary=None, delta_s=None, overlap_class=None,
            note="direct full-FK sample; reference label only",
        ))

    # 4. Small pre-fixed path subset for Link2/Link3. Start must be reference-separated.
    path_rows = []
    path_tasks = []
    path_meta = []
    for arm in RULE["path"]["pairs"]:
        seeds = [row for row in direct_rows if row["arm"] == arm and row["reference_status"] == "exact"
                 and not row["reference_contact"] and row["reference_distance_m"] is not None
                 and 0.001 <= row["reference_distance_m"] <= 0.020]
        seeds.sort(key=lambda row: row["reference_distance_m"])
        selected = []
        for row in seeds:
            if all(np.max(np.abs(np.asarray(row["q"]) - np.asarray(other["q"]))) >= RULE["exclusions"]["min_linf_rad"]
                   for other in selected):
                selected.append(row)
            if len(selected) >= int(RULE["path"]["seeds_per_pair"]):
                break
        for seed_index, seed in enumerate(selected):
            q0 = np.asarray(seed["q"], dtype=np.float64)
            for direction_index in range(int(RULE["path"]["directions_per_seed"])):
                direction = rng.uniform(-1.0, 1.0, 6)
                direction /= max(float(np.abs(direction).max()), 1e-9)
                duration = float(RULE["path"]["max_abs_dq_rad"])
                grid = np.linspace(0.0, duration, int(RULE["path"]["grid_points"]))
                proxy = []
                for s in grid:
                    q = np.clip(q0 + s * direction, LO, HI)
                    set_q(q)
                    proxy.append(screen_distance(arm))
                proxy = np.asarray(proxy)
                contact = np.flatnonzero(proxy <= 0.0)
                if len(contact) and int(contact[0]) > 0:
                    boundary_index = int(contact[0])
                    s_b = float(grid[boundary_index])
                    s_values = [0.0, 0.5 * s_b, 0.8 * s_b, 0.95 * s_b, s_b,
                                min(duration, s_b + 0.05), min(duration, s_b + 0.15)]
                else:
                    boundary_index = int(np.argmin(proxy))
                    s_b = None
                    s_values = [0.0, 0.5 * float(grid[boundary_index]), float(grid[boundary_index]), duration]
                # s=0 is the reference-separated seed; it is kept once as the seed row.
                s_values = sorted({round(float(s), 7) for s in s_values if s > 0.0})
                for s in s_values:
                    q = np.clip(q0 + s * direction, LO, HI)
                    path_id = f"P{arm.rsplit('Link', 1)[1]}_{seed_index}"
                    path_meta.append(dict(arm=arm, seed_row=seed["row_id"], path_id=path_id,
                                          direction=direction_index, s=float(s), s_boundary=s_b,
                                          q=q.copy()))
                print(f"path {arm} seed {seed['row_id']} dir {direction_index}: s_b {s_b}", flush=True)
    # One reference query per path point, still without C/D.
    remaining = max_reference - len(tasks)
    if len(path_meta) > remaining:
        print(f"truncating path tasks {len(path_meta)} -> {remaining} (fixed reference budget)", flush=True)
        path_meta = path_meta[:remaining]
    for index, meta in enumerate(path_meta):
        path_tasks.append((index, meta["arm"], meta["q"], meta))
    path_results = []
    if path_tasks:
        with mp.get_context("fork").Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(reference_task, path_tasks):
                path_results.append(result)
    path_results.sort(key=lambda item: item["task_id"])
    for item in path_results:
        meta = item["meta"]
        result = item["reference"]
        delta = None if meta["s_boundary"] is None else float(meta["s"] - meta["s_boundary"])
        overlap_class = None
        if result["contact"]:
            if delta is None:
                overlap_class = "direct_path_overlap"
            elif delta <= 0.05:
                overlap_class = "path_shallow"
            else:
                overlap_class = "path_deeper"
        path_rows.append(dict(
            row_id=f"P{len(path_rows):04d}", pair=f"{item['arm']} {LIDAR}", arm=item["arm"],
            sample_type="path", source=f"seed:{meta['seed_row']}",
            q=item["q"], cluster=None, screen_mujoco_m=None, screen_band="path",
            reference_status=result["status"], reference_contact=bool(result["contact"]),
            reference_distance_m=result["distance"], reference_lower_bound_m=result["lower_bound"],
            reference_threshold_m=RULE["reference"]["cap_m"], reference_nearest=result["nearest"],
            reference_detail=result["detail"], bin=bin_of(result),
            path_id=meta["path_id"], direction=meta["direction"], s=meta["s"],
            s_boundary=meta["s_boundary"], delta_s=delta, overlap_class=overlap_class,
            note="path parameter is not penetration depth; start was reference-separated",
        ))

    rows = direct_rows + path_rows
    # Coverage accounting. A sample may be in the frozen set more than once only if it is a
    # distinct path point; direct duplicates are already excluded by the L-inf rule.
    coverage = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        bins = {}
        for label in list(RULE["bins_m"]) + ["overlap", ">=30mm_exact", ">=30mm_certified_culled", "failed"]:
            bins[label] = sum(1 for row in subset if row["bin"] == label)
        coverage[arm] = dict(n=len(subset), bins=bins,
                             overlap=sum(1 for row in subset if row["reference_contact"]),
                             exact=sum(1 for row in subset if row["reference_status"] == "exact"),
                             certified_culled=sum(1 for row in subset if row["reference_status"] == "certified_culled"),
                             failed=sum(1 for row in subset if row["reference_status"] == "failed"),
                             sample_types={t: sum(1 for row in subset if row["sample_type"] == t)
                                           for t in sorted({row["sample_type"] for row in subset})})
    record = dict(
        frozen_before_evaluation=True,
        rule_sha256=sha256(RULE_PATH),
        script_sha256=sha256(Path(__file__)),
        input_hashes=RULE["input_hashes"],
        seed=RULE["seed"],
        reference_cap_m=RULE["reference"]["cap_m"],
        contact_tolerance_m=CONTACT_TOL,
        budget=RULE["budget"],
        used_reference_queries=len(tasks) + len(path_tasks),
        random_mujoco_queries=int(RULE["random_joint_samples"]) * len(ARMS),
        joint_limits=dict(low=LO.tolist(), high=HI.tolist()),
        b_class="not used",
        rows=rows,
        coverage=coverage,
        notes=[
            "C/D are not evaluated in this file; the set is frozen before that evaluation.",
            "screen is MuJoCo hull distance, proposal only; reference labels every sample.",
            "reference cap 0.03 m is a certified lower bound; culled rows are not exact distances.",
            "a negative MuJoCo distance is hull overlap, not penetration depth.",
            "path s is a joint-space displacement, not a penetration depth.",
        ],
    )
    (out_dir / "near_set.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(coverage, indent=1))
    print(f"wrote {out_dir / 'near_set.json'}", flush=True)


if __name__ == "__main__":
    main()
