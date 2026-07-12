"""PENGWIN 2026 Task 3 (PENGWIN-Reduction) — 고전 ICP 골절정합(reduction) 백엔드.

문제: 이미 분리된 골절 조각 메시(OBJ 1개, 여러 `g`/`o` 그룹)를 **해부학적 위치로 재조립**하는
6-DoF 강체변환을 조각별로 추정한다. 세그멘테이션이 아니라 **3D 강체 정합/조립(assembly)**이다.

이 모듈은 **numpy/scipy만** 사용하는 고전 ICP 베이스라인이다(무거운 open3d/torch 불필요):

  load_fragment_vertices(obj) → {frag_id: (N,3)}   OBJ 파싱(스트리밍, v + g그룹 face→vertex 집합)
  kabsch(P, Q)               → (R, t)              대응점 P→Q 최소자승 강체정합(SVD, det 반사보정)
  icp(src, dst)              → (R, t, rmse)        cKDTree 최근접대응 + kabsch 반복
  reduce_fragments(frags)    → {frag_id: 4x4}      greedy 조립(SA1=identity anchor, 나머지 ICP-to-union)

설계 원칙 — **identity가 안전 바닥**:
  평가 규약상 어떤 조각의 결과를 못 내면 그 변환은 Identity로 간주(= 감점 하한). 따라서
  reduce_fragments 는 (1) 모든 입력 조각에 대해 반드시 항목을 채우고, (2) ICP가 조각을 오히려
  **나쁘게** 옮길 위험이 있으면 그 조각을 identity 로 되돌린다. 구체적 가드:
    - ICP 결과가 union 포인트클라우드에 대한 **평균 최근접거리(mean-NN)를 낮출 때만** 채택,
    - 회전각 > max_rot_deg 또는 조각 중심 이동 > max_trans_mm 이면 기각 → identity,
    - 조각 처리 중 어떤 예외든 발생하면 그 조각만 identity 로 처리(전체가 죽지 않음).

조각 ID 규약(⚠️ Task1/2의 라벨 1–200과 다름):
    1–100  Sacrum (SA) ,  101–200  Left Ilium (LI) ,  201–300  Right Ilium (RI)   (femur 없음)
SA 조각 1이 있으면 anchor 로 고정(=identity). 조직위 평가가 모든 포즈를 SA1 기준으로 재표현하므로
SA1=identity 출력이 규약에 자동 부합한다.
"""
from __future__ import annotations

import numpy as np

try:  # scipy 는 정상 환경/컨테이너에 항상 존재. 부재 시 상위 호출부가 identity fallback.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


IDENTITY_4x4 = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]


def _identity():
    """공유 mutable 실수 방지용 — 매번 새 identity 4x4 리스트를 만든다."""
    return [row[:] for row in IDENTITY_4x4]


def _first_int(s):
    """문자열에서 첫 정수 토큰을 파싱(없으면 None). `101`, `LI_101`, `fragment 201` 모두 처리."""
    cur = ""
    for ch in s:
        if ch.isdigit():
            cur += ch
        elif cur:
            break
    return int(cur) if cur else None


# --------------------------------------------------------------------------- #
# OBJ I/O                                                                      #
# --------------------------------------------------------------------------- #
def load_fragment_vertices(obj_path):
    """OBJ → {frag_id(int): np.ndarray(N,3)} 조각별 정점 집합.

    - `v x y z` = 전역 정점(1-기반). 스트리밍으로 읽어 대용량 메시에 대비.
    - `g <id>` / `o <id>` = 그룹/오브젝트. 라인의 첫 정수를 조각 ID로 사용(`g fragment_101` 등 허용).
    - `f a/.. b/.. c/..` = 면. 각 토큰의 첫 정수(정점 인덱스, 음수 상대참조 지원)를 **현재 그룹**에
      귀속시켜 조각별 정점 집합을 만든다(공유 정점풀에서도 올바르게 조각을 분리).
    - **폴백**: 면이 전혀 없는 그룹은, 그 그룹이 current 인 동안 선언된 정점들을 조각으로 귀속한다
      (일부 익스포터가 조각을 `g`+`v`만으로 내보내는 경우 대비).

    Returns: {frag_id: (N,3) float64}. 정점이 하나도 없는 그룹은 생략된다.
    """
    verts = []          # [(x,y,z), ...] 전역 정점
    vert_group = []     # 각 정점 선언 시점의 current 그룹(폴백용), verts 와 평행
    face_group = {}     # gid -> set(0-based vertex index)  (면 기반 귀속)
    cur = None

    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            c0 = line[0]
            if c0 == "v":
                # 'v ' 만 정점; 'vn'/'vt'/'vp' 는 제외
                if len(line) > 1 and (line[1] == " " or line[1] == "\t"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                            vert_group.append(cur)
                        except ValueError:
                            pass
            elif c0 == "g" or c0 == "o":
                cur = _first_int(line[1:])
                if cur is not None:
                    face_group.setdefault(cur, set())
            elif c0 == "f":
                if len(line) > 1 and (line[1] == " " or line[1] == "\t") and cur is not None:
                    bucket = face_group.setdefault(cur, set())
                    for tok in line.split()[1:]:
                        vs = tok.split("/", 1)[0]
                        try:
                            vi = int(vs)
                        except ValueError:
                            continue
                        # OBJ 인덱스: 양수=1기반 절대, 음수=현재까지 정점수 기준 상대(-1=마지막)
                        vi = (len(verts) + vi) if vi < 0 else (vi - 1)
                        bucket.add(vi)

    V = np.asarray(verts, dtype=np.float64) if verts else np.zeros((0, 3), dtype=np.float64)
    n = V.shape[0]

    frags = {}
    for gid, vset in face_group.items():
        if gid is None or not vset:
            continue
        idx = np.fromiter((i for i in vset if 0 <= i < n), dtype=np.int64, count=-1)
        if idx.size:
            frags[gid] = V[idx]

    # 폴백: 면 없는(또는 비어있는) 그룹은 선언-귀속 정점으로 채운다.
    decl = {}
    for i, g in enumerate(vert_group):
        if g is not None:
            decl.setdefault(g, []).append(i)
    for gid, idxs in decl.items():
        if gid not in frags:
            arr = V[np.asarray(idxs, dtype=np.int64)]
            if arr.shape[0] > 0:
                frags[gid] = arr

    return frags


def _to_4x4(R, t):
    """(R 3x3, t 3,) → 4x4 row-major homogeneous(list of list of float). x' = R x + t."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M.tolist()


# --------------------------------------------------------------------------- #
# Rigid registration                                                          #
# --------------------------------------------------------------------------- #
def kabsch(P, Q):
    """대응점 최소자승 강체정합: R P + t ≈ Q 를 최소화하는 (R, t) 반환.

    Kabsch/Umeyama(회전만; 스케일 없음). SVD 후 det 부호로 반사(개선판 반사행렬)를 제거해
    항상 **proper rotation**(det=+1)을 보장한다. P, Q 는 (N,3), 대응 순서 일치 가정.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    Pc = P - cP
    Qc = Q - cQ
    H = Pc.T @ Qc                      # 3x3 교차공분산
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T                 # P -> Q 회전
    t = cQ - R @ cP
    return R, t


def rotation_angle_deg(R):
    """회전행렬 → 측지 회전각(도). trace 기반, 수치오차로 arccos 인자는 [-1,1]로 클립."""
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, float(c)))
    return float(np.degrees(np.arccos(c)))


def _subsample(pts, max_points, rng):
    """포인트클라우드를 max_points 이하로 무작위 서브샘플(고정 seed 로 재현성). None=무샘플."""
    n = pts.shape[0]
    if max_points is None or n <= max_points:
        return pts
    idx = rng.choice(n, size=max_points, replace=False)
    return pts[idx]


def _mean_nn(pts, tree):
    """pts 각 점의 tree(dst) 최근접거리 평균."""
    d, _ = tree.query(pts)
    return float(np.mean(d))


def icp(src_pts, dst_pts, max_iter=30, tol=1e-4, max_points=20000, seed=0):
    """고전 point-to-point ICP: cKDTree 최근접대응 + kabsch 반복.

    src_pts 를 dst_pts 에 정합하는 누적 강체변환 (R, t) 와 최종 rmse 를 반환한다
    (src 를 옮긴 결과 x' = R x + t 가 dst 에 밀착). 재현성을 위해 서브샘플은 고정 seed.

    Returns: (R 3x3, t (3,), rmse float). scipy 부재 시 identity 변환 + inf.
    """
    if cKDTree is None:
        return np.eye(3), np.zeros(3), float("inf")

    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s = _subsample(src, max_points, rng)
    d = _subsample(dst, max_points, rng)

    tree = cKDTree(d)
    R_tot = np.eye(3, dtype=np.float64)
    t_tot = np.zeros(3, dtype=np.float64)
    cur = s.copy()
    prev = np.inf
    rmse = np.inf

    for _ in range(int(max_iter)):
        dists, idx = tree.query(cur)
        R, t = kabsch(cur, d[idx])
        cur = cur @ R.T + t
        # 누적: x_new = R (R_tot x + t_tot) + t
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        if abs(prev - rmse) < tol:
            break
        prev = rmse

    # 마지막 갱신 이후의 rmse 로 최종화
    dists, _ = tree.query(cur)
    rmse = float(np.sqrt(np.mean(dists ** 2)))
    return R_tot, t_tot, rmse


# --------------------------------------------------------------------------- #
# Greedy assembly                                                             #
# --------------------------------------------------------------------------- #
def reduce_fragments(frags, max_rot_deg=60.0, max_trans_mm=80.0,
                     max_iter=30, tol=1e-4, min_points=6,
                     max_points=20000, seed=0):
    """조각별 6-DoF 포즈를 greedy 조립으로 추정 → {frag_id: 4x4 row-major list}.

    절차:
      1. anchor = SA 조각 1(있으면), 없으면 가장 큰 조각. anchor 는 identity.
      2. 나머지 조각을 크기 내림차순으로, **이미 배치된 조각들의 합집합(union) 포인트클라우드**에
         ICP 정합. union 이 커질수록 mating 면이 늘어 뒤 조각이 더 안정적으로 붙는다.
      3. **가드(identity 바닥)**: ICP 가 union 대한 mean-NN 을 낮출 때만 채택하고,
         회전 > max_rot_deg 또는 중심이동 > max_trans_mm 면 기각 → 그 조각 identity.
      4. 조각 처리 중 예외/정점부족(min_points 미만)이면 그 조각만 identity.

    **모든 입력 조각에 대해 반드시 항목을 포함**한다(평가 규약: 누락 시 identity 처벌이지만 명시 출력).

    Args:
      max_rot_deg / max_trans_mm : 채택 상한(도/mm). 이 이상은 catastrophic 이동으로 보고 기각.
    """
    result = {}
    if not frags:
        return result

    ids = list(frags.keys())

    # anchor 선택: SA1(id==1) 우선, 없으면 최대 정점 조각.
    if 1 in frags and frags[1].shape[0] >= min_points:
        anchor = 1
    else:
        anchor = max(ids, key=lambda k: frags[k].shape[0])

    result[anchor] = _identity()
    placed = [np.asarray(frags[anchor], dtype=np.float64)]

    remaining = sorted((k for k in ids if k != anchor),
                       key=lambda k: -frags[k].shape[0])

    for fid in remaining:
        try:
            src = np.asarray(frags[fid], dtype=np.float64)
            if src.shape[0] < min_points or cKDTree is None:
                result[fid] = _identity()
                placed.append(src)
                continue

            dst = np.concatenate(placed, axis=0)
            R, t, _rmse = icp(src, dst, max_iter=max_iter, tol=tol,
                              max_points=max_points, seed=seed)

            # 가드용 mean-NN (동일 서브샘플 seed 로 tree 재구성 — 채택/기각 판정의 결정성 확보)
            rng = np.random.default_rng(seed)
            d_sub = _subsample(dst, max_points, rng)
            tree = cKDTree(d_sub)
            before = _mean_nn(src, tree)
            src_moved = src @ R.T + t
            after = _mean_nn(src_moved, tree)

            rot = rotation_angle_deg(R)
            disp = float(np.linalg.norm(src_moved.mean(axis=0) - src.mean(axis=0)))

            if (after < before) and (rot <= max_rot_deg) and (disp <= max_trans_mm):
                result[fid] = _to_4x4(R, t)
                placed.append(src_moved)
            else:
                result[fid] = _identity()
                placed.append(src)
        except Exception:
            # 이 조각만 안전 바닥으로. union 에는 원위치로 넣는다.
            result[fid] = _identity()
            try:
                placed.append(np.asarray(frags[fid], dtype=np.float64))
            except Exception:
                pass

    return result
