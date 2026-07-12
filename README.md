# PENGWIN 2026 — Task 3 (PENGWIN-Reduction) 알고리즘 컨테이너

**골반 골절 정합(reduction planning)** 을 위한 **경량 고전 ICP** 컨테이너입니다.
분할(segmentation)이 아니라 **3D 강체 정합/조립**이므로 딥러닝/GPU/nnUNet 이 필요 없고,
`python:3.10-slim` + `numpy/scipy` 만으로 동작합니다(이미지 수백 MB).

## 무엇을 하는가
- 입력: 여러 골절 조각 메시가 담긴 **OBJ 1개** (`g`/`o` 그룹 ID 로 조각 구분).
  - ID 규약: `1–100 Sacrum(SA)`, `101–200 Left Ilium(LI)`, `201–300 Right Ilium(RI)` — Task 1/2 와 다름.
- 출력: 조각별 **4×4 row-major 강체변환**(재조립 포즈)을 담은 **JSON**.
- 규약: **SA 조각 1 = identity anchor**(평가가 모든 포즈를 SA1 기준으로 재표현).

## 방법 — 고전 ICP greedy 조립 (`inference/reduction.py`)
1. `load_fragment_vertices` — OBJ 스트리밍 파싱(정점 + `g`그룹 면→정점 집합, 폴백: 선언-귀속).
2. `kabsch(P,Q)` — 대응점 최소자승 강체정합(SVD + det 반사보정 → proper rotation).
3. `icp(src,dst)` — `scipy.spatial.cKDTree` 최근접대응 + kabsch 반복(누적 R,t, rmse).
4. `reduce_fragments` — **greedy 조립**: SA1(있으면)=identity anchor, 나머지 조각은 크기 내림차순으로
   **이미 배치된 조각들의 union 포인트클라우드**에 ICP.

### identity = 안전 바닥 (guard)
평가 규약상 결과를 못 내면 그 변환은 **Identity** 로 간주(감점 하한). 이를 적극 활용해:
- ICP 가 union 대한 **평균 최근접거리(mean-NN)를 낮출 때만** 채택,
- 회전 > `max_rot_deg`(기본 60°) 또는 조각 중심 이동 > `max_trans_mm`(기본 80mm) 이면 기각 → identity,
- 조각 처리 중 예외/정점 부족이면 그 조각만 identity(전체가 죽지 않음).

즉 ICP 가 조각을 **오히려 나쁘게** 옮길 위험이 있으면 항상 identity 로 되돌려 바닥을 지킵니다.

## Grand Challenge I/O (`--network none`)
| Path | Role |
|---|---|
| `/input/peripelvic-fracture-fragments-meshes.obj` | 읽기전용 입력 OBJ |
| `/output/reduction-poses-matrices.json` | 예측 포즈 JSON(쓰기) |
| `/opt/ml/model/` | 모델 tarball(이 베이스라인은 미사용) |

출력 JSON 예:
```json
[
  {"fragment_id": "1",   "transformation": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  {"fragment_id": "101", "transformation": [[0.98,0.04,-0.19,-24.7],[...],[...],[0,0,0,1]]}
]
```
**입력의 모든 조각에 대해 항목을 반드시 포함**합니다.

## 빌드 & 로컬 실행
```bash
# 이미지 빌드(빌드 컨텍스트 = 저장소 루트)
IMAGE_TAG=pengwin-task3-reduction:latest scripts/build_image.sh

# 로컬 파이썬 실행(컨테이너 없이)
python inference/inference.py <mesh.obj> [out.json]

# 컨테이너 실행(GC 규약 흉내)
docker run --rm --network none \
    -v /path/to/case:/input:ro \
    -v /path/to/out:/output \
    pengwin-task3-reduction:latest
```

## 레이아웃
```
inference/
  inference.py     ENTRYPOINT — OBJ 읽기 → reduce → JSON, 실패 시 전 조각 identity fallback
  reduction.py     고전 ICP greedy 조립 백엔드(numpy/scipy, self-contained)
  __init__.py
Dockerfile         python:3.10-slim, 비root user, HOME=/tmp
requirements.txt   numpy / scipy / trimesh (핀)
scripts/build_image.sh
LICENSE            MIT
```

## 컨테이너 규약(Task 1 미러)
- 비root: `groupadd -r user && useradd -r -g user user; USER user:user`.
- `HOME=/tmp`, `MPLCONFIGDIR=/tmp/matplotlib`(비root 쓰기권한 회피).
- `ENTRYPOINT ["python","/opt/app/inference/inference.py"]`; `COPY inference /opt/app/inference`.

## 다음 단계
현재는 고전 ICP 베이스라인입니다. 학습형 pose regression(FracFormer 계열, 480×60 시뮬레이션 세트)로
`reduce_fragments`/`predict_poses` 를 교체하면 되고, I/O·조각 열거·JSON 포맷은 이미 계약대로 맞춰져
있습니다. Official baseline: https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline
