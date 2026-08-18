#!/usr/bin/env python3
"""PENGWIN 2026 Task 3 (Reduction) — learned AssemblyTransformer GC entrypoint.

Reads one OBJ from /input, runs the vendored official baseline (baseline/inference.py) with the trained
checkpoint at /opt/ml/model/model.ckpt, writes /output/reduction-poses-matrices.json.
On ANY failure falls back to all-identity poses (the verified score floor) so the container never crashes.
"""
from __future__ import annotations
import glob, json, os, shutil, subprocess, sys
from pathlib import Path

IN_OBJ  = os.environ.get("PENGWIN_INPUT_OBJ")  or None
OUT_JSON = os.environ.get("PENGWIN_OUTPUT_JSON", "/output/reduction-poses-matrices.json")
APP = Path(__file__).resolve().parent.parent          # /opt/app
BASE = APP / "baseline"

def _find_obj():
    if IN_OBJ and os.path.exists(IN_OBJ):
        return IN_OBJ
    for pat in ("/input/*.obj", "/input/**/*.obj", "/input/*fragments*meshes*.obj"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None

def _parse_frag_ids(obj_path):
    ids = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("g "):
                name = line[2:].strip()
                if name and name not in ids:
                    ids.append(name)
    return ids or ["1"]

def _identity(obj_path):
    ids = _parse_frag_ids(obj_path) if obj_path else ["1"]
    I = [[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,0,1.0]]
    return [{"fragment_id": fid, "transformation": I} for fid in ids]

def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    obj = _find_obj()
    poses = None
    try:
        assert obj is not None, "no input OBJ found"
        work = Path("/tmp/t3in/case"); shutil.rmtree("/tmp/t3in", ignore_errors=True); work.mkdir(parents=True)
        shutil.copy(obj, work / "frags.obj")
        r = subprocess.run(
            [sys.executable, str(BASE / "inference.py"), "--config", os.environ.get("PENGWIN_T3_CONFIG","configs/test_gc.yaml"), "--input_dir", "/tmp/t3in"],
            cwd=str(BASE), capture_output=True, text=True, timeout=540,
        )
        # 최종 full/coords 설정은 기본 파일명을 쓰지만 result_suffix가 붙어도 결과를 잃지 않도록
        # 같은 prefix의 JSON을 찾는다. 출력이 없을 때만 아래 identity 형식 fallback으로 간다.
        outs = sorted(work.glob("reduction-poses-matrices*.json"))
        if outs:
            with outs[0].open() as f:
                poses = json.load(f)
        else:
            sys.stderr.write(f"[t3] baseline produced no output (rc={r.returncode}); stderr tail:\n{r.stderr[-2000:]}\n")
    except Exception as exc:
        sys.stderr.write(f"[t3] learned inference failed ({exc}) -> identity fallback\n")
    if not poses:
        poses = _identity(obj)
    with open(OUT_JSON, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"[t3] wrote {len(poses)} poses -> {OUT_JSON}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
