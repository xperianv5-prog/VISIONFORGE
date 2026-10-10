"""VisionForge episodic memory (SQLite)."""
import os
import json
import sqlite3
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("VF_MEMORY_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "visionforge_memory.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    job_id TEXT,
    input_hash TEXT NOT NULL,
    object_class TEXT NOT NULL,
    backend TEXT,
    target_faces INTEGER,
    quad_ratio REAL,
    non_manifold_edges INTEGER,
    bake_resolution INTEGER,
    bake_maps TEXT,
    cage_extrusion REAL,
    ssim REAL,
    lpips REAL,
    silhouette_iou REAL,
    qa_score REAL,
    qa_pass INTEGER,
    packed_ok INTEGER,
    label TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    provider TEXT,
    duration_s REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_class ON runs(object_class);
CREATE INDEX IF NOT EXISTS idx_runs_hash ON runs(input_hash);
"""

DEFAULTS = {"target_faces": 12000, "bake_resolution": 4096, "cage_extrusion": 0.02, "backend": "triposr"}


@contextmanager
def _conn(path: str = DB_PATH):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: str = DB_PATH) -> None:
    with _conn(path) as con:
        con.executescript(SCHEMA)


def record_run(rec: Dict[str, Any], path: str = DB_PATH) -> int:
    init_db(path)
    cols = ["job_id", "input_hash", "object_class", "backend", "target_faces", "quad_ratio", "non_manifold_edges",
            "bake_resolution", "bake_maps", "cage_extrusion", "ssim", "lpips", "silhouette_iou", "qa_score",
            "qa_pass", "packed_ok", "label", "prompt_tokens", "completion_tokens", "provider", "duration_s", "notes"]
    row = {c: rec.get(c) for c in cols}
    if isinstance(row["bake_maps"], (list, tuple)):
        row["bake_maps"] = json.dumps(list(row["bake_maps"]))
    row["qa_pass"] = int(bool(row["qa_pass"])) if row["qa_pass"] is not None else None
    row["packed_ok"] = int(bool(row["packed_ok"])) if row["packed_ok"] is not None else None
    row["input_hash"] = row["input_hash"] or "unknown"
    row["object_class"] = (row["object_class"] or "object").lower()
    with _conn(path) as con:
        cur = con.execute(
            f"INSERT INTO runs (created_at,{','.join(cols)}) VALUES (?,{','.join('?' for _ in cols)})",
            [time.time()] + [row[c] for c in cols])
        return int(cur.lastrowid)


def query_runs(object_class: Optional[str] = None, input_hash: Optional[str] = None, only_success: bool = False,
               limit: int = 10, path: str = DB_PATH) -> List[Dict[str, Any]]:
    init_db(path)
    sql = "SELECT * FROM runs WHERE 1=1"
    args: List[Any] = []
    if object_class:
        sql += " AND object_class = ?"
        args.append(object_class.lower())
    if input_hash:
        sql += " AND input_hash = ?"
        args.append(input_hash)
    if only_success:
        sql += " AND qa_pass = 1 AND packed_ok = 1"
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with _conn(path) as con:
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def recommend(object_class: str, animation_candidate: bool = False, path: str = DB_PATH) -> Dict[str, Any]:
    """Few-shot episodic retrieval: derive optimal remesh/bake settings from successful past runs."""
    runs = query_runs(object_class=object_class, only_success=True, limit=25, path=path)
    source = "class"
    if len(runs) < 2:
        runs = query_runs(only_success=True, limit=50, path=path)
        source = "global" if runs else "defaults"
    rec = dict(DEFAULTS)
    rec["source"] = source
    rec["examples"] = len(runs)
    if runs:
        faces = [r["target_faces"] for r in runs if r["target_faces"]]
        res = [r["bake_resolution"] for r in runs if r["bake_resolution"]]
        cages = [r["cage_extrusion"] for r in runs if r["cage_extrusion"]]
        backends = [r["backend"] for r in runs if r["backend"]]
        if faces:
            rec["target_faces"] = int(statistics.median(faces))
        if res:
            rec["bake_resolution"] = Counter(res).most_common(1)[0][0]
        if cages:
            rec["cage_extrusion"] = round(float(statistics.median(cages)), 4)
        if backends:
            rec["backend"] = Counter(backends).most_common(1)[0][0]
    if animation_candidate:
        rec["target_faces"] = max(rec["target_faces"], 20000)
        rec["use_mesh_symmetry"] = True
    else:
        rec["use_mesh_symmetry"] = False
    rec["target_faces"] = int(min(max(rec["target_faces"], 2000), 80000))
    return rec


def few_shot_examples(object_class: str, k: int = 3, path: str = DB_PATH) -> str:
    """Compact text block of past successful runs for System 2 prompts."""
    runs = query_runs(object_class=object_class, only_success=True, limit=k, path=path)
    if not runs:
        runs = query_runs(only_success=True, limit=k, path=path)
    lines = []
    for r in runs:
        lines.append(f"{r['object_class']}:faces={r['target_faces']},quad={r['quad_ratio']},res={r['bake_resolution']},"
                     f"qa={r['qa_score']},label={r['label']}")
    return ";".join(lines)


def stats(path: str = DB_PATH) -> Dict[str, Any]:
    init_db(path)
    with _conn(path) as con:
        total = con.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"]
        passed = con.execute("SELECT COUNT(*) c FROM runs WHERE qa_pass=1").fetchone()["c"]
        tokens = con.execute("SELECT COALESCE(SUM(prompt_tokens),0) p, COALESCE(SUM(completion_tokens),0) c FROM runs").fetchone()
        labels = con.execute("SELECT label, COUNT(*) c FROM runs GROUP BY label").fetchall()
    return {"runs": total, "qa_passed": passed, "prompt_tokens": tokens["p"], "completion_tokens": tokens["c"],
            "labels": {r["label"]: r["c"] for r in labels}}


if __name__ == "__main__":
    init_db()
    rid = record_run({"job_id": "demo", "input_hash": "abc", "object_class": "chair", "backend": "triposr",
                      "target_faces": 15000, "quad_ratio": 0.97, "bake_resolution": 4096, "cage_extrusion": 0.02,
                      "ssim": 0.81, "qa_score": 0.84, "qa_pass": True, "packed_ok": True, "label": "STATIC_READY"})
    print("inserted", rid)
    print(recommend("chair"))
    print(stats())
3 · Headless Blender Automation Engine — pipeline.py
Run as blender -b --python pipeline.py -- --job job.json. Works on 3.6 LTS and 4.x: socket lookup goes through an alias table (Emission→Emission Color, Specular→Specular IOR Level, …), importers fall back between wm.obj_import and legacy operators, and Eevee's renamed engine id is handled. Metallic and curvature are baked through the Emission channel (Cycles has no native pass for them); curvature uses the Geometry node's Pointiness output.

pipeline.py
visionforge/pipeline.py
"""VisionForge headless Blender pipeline. Runs INSIDE Blender (3.6 LTS and 4.x).
Usage: blender -b --python pipeline.py -- --job job.json
"""
import bpy
import bmesh
import sys
import os
import json
import math
import time
import subprocess
from mathutils import Vector

VERSION = bpy.app.version
IS_4X = VERSION >= (4, 0, 0)
EEVEE_ID = "BLENDER_EEVEE_NEXT" if VERSION >= (4, 2, 0) else "BLENDER_EEVEE"

SOCKET_ALIASES = {
    "Specular": ["Specular IOR Level", "Specular"],
    "Subsurface": ["Subsurface Weight", "Subsurface"],
    "Emission": ["Emission Color", "Emission"],
    "Emission Strength": ["Emission Strength"],
    "Transmission": ["Transmission Weight", "Transmission"],
    "Clearcoat": ["Coat Weight", "Clearcoat"],
    "Sheen": ["Sheen Weight", "Sheen"],
    "Base Color": ["Base Color"],
    "Metallic": ["Metallic"],
    "Roughness": ["Roughness"],
    "Normal": ["Normal"],
    "Alpha": ["Alpha"],
}


def log(msg):
    print(f"[VF {time.strftime('%H:%M:%S')}] {msg}", flush=True)
