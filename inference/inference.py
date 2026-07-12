"""PENGWIN 2026 Task 3 (PENGWIN-Reduction) — 컨테이너 추론 진입점.

문제: **분할이 아니라 3D 강체 정합/조립(reduction planning).** 입력은 이미 분리된 골절 조각
메시(OBJ 1개, 여러 `g` 그룹), 출력은 조각별 **4×4 강체 변환행렬(JSON)**.

이 파일이 구현하는 것 = **identity 베이스라인**(= 안전 바닥). 근거:
  - 평가 규약: *"알고리즘이 어떤 케이스의 결과를 못 내면 변환행렬은 **Identity**로 간주된다."*
    (`docs/challenge/03-task3-reduction-planning.md`) → identity는 항상 유효하고 감점의 하한이다.
  - 평가는 SA 조각 1(Anchor)을 identity로 고정하고 나머지를 **상대적으로만** 본다.
    조직위 권장: "SA fragment 1을 identity로 출력하라." → 전부 identity면 이 규약에 자동 부합.

이 베이스라인은 **점수 바닥을 확보**하고 컨테이너 I/O 계약을 검증하는 용도다. 실제 reduction
모델(ICP + 학습형 pose regression, baseline repo Sutuk/PENGWIN2026_Task3_Reduction_Baseline)은
`predict_poses()`를 교체해 얹으면 된다 — I/O·조각 열거·JSON 포맷은 이미 계약대로 맞춰져 있다.

조각 ID 규약(⚠️ Task1/2의 라벨 1–200과 다름):
    1–100  Sacrum (SA) ,  101–200  Left Ilium (LI) ,  201–300  Right Ilium (RI)   (femur 없음)

컨테이너 경로(GC, `--network none`):
    /input/peripelvic-fracture-fragments-meshes.obj   (읽기전용)
    /output/reduction-poses-matrices.json             (쓰기)
    /opt/ml/model/                                     (모델 tarball, 읽기전용)
로컬 테스트: `python inference.py <mesh.obj> [out.json]`.
"""
from __future__ import annotations
import json
import os
import sys

# GC 컨테이너 기본 경로 (guidelines Submission 표)
DEFAULT_INPUT_OBJ = "/input/peripelvic-fracture-fragments-meshes.obj"
DEFAULT_OUTPUT_JSON = "/output/reduction-poses-matrices.json"

IDENTITY_4x4 = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]


def parse_fragment_ids(obj_path):
    """OBJ에서 조각 ID 목록을 **입력 등장 순서**대로 뽑는다.

    조각 정체성은 `g <id>` (또는 `o <id>`) 그룹/오브젝트 라인의 정수 ID로 인코딩된다.
    ID 토큰에 접두어가 섞여 있어도(예: `g fragment_101`) 라인에서 첫 정수를 찾아 쓴다.
    OBJ 전체를 메모리에 올리지 않도록 라인 스트리밍으로 읽는다(대용량 메시 대비).

    Returns: 정수 ID들의 리스트(등장 순서, 중복 제거).
    """
    ids = []
    seen = set()
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            tag = line[:2]
            if tag == "g " or tag == "o ":
                tok = _first_int(line[2:])
                if tok is not None and tok not in seen:
                    seen.add(tok)
                    ids.append(tok)
    return ids


def _first_int(s):
    """문자열에서 첫 정수 토큰을 파싱(없으면 None). `101`, `LI_101`, `fragment 201` 모두 처리."""
    cur = ""
    for ch in s:
        if ch.isdigit():
            cur += ch
        elif cur:
            break
    return int(cur) if cur else None


def anatomy_of(frag_id):
    """조각 ID → 해부학 약어 (SA/LI/RI). 범위 밖이면 'UNK'. (진단·로깅용)"""
    if 1 <= frag_id <= 100:
        return "SA"
    if 101 <= frag_id <= 200:
        return "LI"
    if 201 <= frag_id <= 300:
        return "RI"
    return "UNK"


def predict_poses(obj_path):
    """조각별 6-DoF 포즈를 예측한다. **고전 ICP greedy 조립(reduction.reduce_fragments).**

    입력의 모든 조각에 대해 항목을 반드시 포함해야 한다(평가 규약). reduction 백엔드가 어떤 이유로든
    import/실행에 실패하면 **전 조각 identity** 로 안전하게 되돌아간다(= 감점 하한). 개별 조각의
    실패는 reduce_fragments 내부에서 이미 그 조각 identity 로 처리된다.

    Returns: [{"fragment_id": "<id>", "transformation": <4x4 row-major>}...]
    """
    ids = parse_fragment_ids(obj_path)

    poses_by_id = {}
    # ⚠️ ICP는 기본 OFF (검증된 결론, 2026-07-12).
    #    클리니컬 GT(`plan_pl_gt.json`, 170케이스)로 실측: greedy overlap-ICP가 identity보다 **나쁘다**
    #    (Rot 18.1° vs 9.5°, Trans 66mm vs 46mm). reduction은 골절면 **mating**이지 overlap이 아니라서
    #    ICP-to-union이 조각을 겹치게 당겨 방향이 어긋난다. fail→identity가 평가 바닥이고 identity가 ICP를
    #    이기므로 **컨테이너는 검증된 바닥 identity를 기본 출력**한다. ICP는 `PENGWIN_T3_ICP=1`일 때만.
    if os.environ.get("PENGWIN_T3_ICP") == "1":
        try:
            # reduction.py 는 같은 디렉터리의 형제 모듈(컨테이너에도 inference/ 에 동봉).
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from reduction import load_fragment_vertices, reduce_fragments
            frags = load_fragment_vertices(obj_path)
            for fid, M in reduce_fragments(frags).items():
                poses_by_id[int(fid)] = M
        except Exception as e:  # 백엔드 전체 실패 → 전 조각 identity fallback
            sys.stderr.write(f"[task3] reduction 백엔드 실패 → identity fallback: {e}\n")
            poses_by_id = {}

    poses = []
    for fid in ids:
        # Anchor 규약: SA 조각 1은 reduce_fragments 가 identity 로 반환(없으면 여기서도 identity).
        poses.append({"fragment_id": str(fid),
                      "transformation": poses_by_id.get(fid, IDENTITY_4x4)})
    # parse 는 못 봤지만 reduction 이 찾은 조각(희귀)도 누락 없이 포함
    for fid, M in poses_by_id.items():
        if fid not in ids:
            poses.append({"fragment_id": str(fid), "transformation": M})
    return poses


def run(input_obj=None, output_json=None):
    """컨테이너 진입: OBJ 읽기 → 포즈 예측 → JSON 쓰기. 실패해도 빈 리스트라도 남긴다."""
    input_obj = input_obj or DEFAULT_INPUT_OBJ
    output_json = output_json or DEFAULT_OUTPUT_JSON

    poses = []
    try:
        poses = predict_poses(input_obj)
    except Exception as e:  # 어떤 이유로든 실패 시: 빈 결과(평가에서 케이스별 identity로 처리됨)
        sys.stderr.write(f"[task3] predict_poses 실패({input_obj}): {e}\n")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(poses, f, indent=2)

    n_by = {}
    n_moved = 0
    for p in poses:
        n_by[anatomy_of(int(p["fragment_id"]))] = n_by.get(anatomy_of(int(p["fragment_id"])), 0) + 1
        if p["transformation"] != IDENTITY_4x4:
            n_moved += 1
    sys.stderr.write(
        f"[task3] {len(poses)}개 조각 포즈 기록(정합 {n_moved} / identity {len(poses) - n_moved}) "
        f"→ {output_json}  ({n_by})\n")
    return poses


if __name__ == "__main__":
    args = sys.argv[1:]
    run(args[0] if len(args) >= 1 else None,
        args[1] if len(args) >= 2 else None)
