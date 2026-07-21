# PENGWIN 2026 — Task 3: 골절 정복 계획 (PENGWIN-Reduction)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10--slim-blue.svg)](https://www.python.org/)
[![deps: numpy+scipy only](https://img.shields.io/badge/deps-numpy%20%2B%20scipy%20only-green.svg)](requirements.txt)
[![no GPU](https://img.shields.io/badge/GPU-not%20required-lightgrey.svg)](Dockerfile)

> **PENGWIN 2026 Grand Challenge — Task 3**: 골절로 분리된 골반 조각 메시들을 **해부학적 정상 위치로 되돌리는 강체 변환(4×4)** 을 예측한다.
>
> 세그멘테이션이 아니라 **3D 강체 정합(assembly)** 문제다. 따라서 이 저장소는 torch·nnU-Net·GPU를 전혀 쓰지 않는다 — 순수 numpy/scipy.

---

## 🚀 현재 배포 상태 (2026-07-21, v1.0)

| | |
|---|---|
| **배포 버전** | **v1.0** — git tag `v1.0` push → GC 자동 빌드 |
| **출력 전략** | **모든 조각에 identity 행렬** (ICP는 `PENGWIN_T3_ICP=1` 일 때만 활성, 기본 OFF) |
| **왜 identity인가** | 구현한 greedy overlap-ICP 와 RF pose-regressor **둘 다 identity보다 나빴다**. 정직한 negative result — §4 |
| **베이스 이미지** | `python:3.10-slim` (GPU 불필요, 이미지 ~120MB) |
| **의존성** | `numpy>=1.24,<3`, `scipy>=1.10,<2` — 그게 전부 |
| **GC 채점 이력** | **아직 없음** — 첫 제출 시 스모크 검증 필요 |

> ℹ️ 공식 규정상 **결과 미제출 케이스는 identity로 간주**된다. 따라서 현재 컨테이너는 "실패 제출"과
> 점수가 같다. 이는 인정된 사실이며, 왜 그럼에도 이것이 옳은 선택인지는 §4·§6에 정량적으로 기록했다.

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [평가 지표](#2-평가-지표)
3. [알고리즘 — 구현된 것](#3-알고리즘--구현된-것)
4. [왜 identity인가 — 정량적 근거](#4-왜-identity인가--정량적-근거)
5. [로컬 GT 게이트](#5-로컬-gt-게이트)
6. [다음 레버](#6-다음-레버)
7. [재현 / 빌드 / 검증](#7-재현--빌드--검증)
8. [부록](#8-부록)

---

## 1. 프로젝트 개요

### 1.1 대회 / 태스크

- **대회**: PENGWIN 2026 Grand Challenge — **Task 3 (PENGWIN-Reduction)**
- **입력**: `pelvic-fracture-fragments` — 여러 메시가 담긴 **단일 OBJ** 파일
- **출력**: `/output/reduction-poses-matrices.json` — 조각별 **4×4 row-major** 변환
- **데이터**: 임상 170 케이스 + 시뮬레이션(60 골절 패턴 × 480 골반 모델)
- **배포 환경**: GC 컨테이너, `--network none`

### 1.2 조각 ID 체계 (Task 1/2와 다름!)

Task 3은 OBJ의 `g` 그룹 이름에서 ID를 얻으며, **대퇴골이 없다**.

| ID 범위 | 해부학 | 약어 |
|---|---|---|
| `1 – 100` | Sacrum (천골) | SA |
| `101 – 200` | Left Ilium (좌장골) | LI |
| `201 – 300` | Right Ilium (우장골) | RI |

### 1.3 Anchor 규약

> **모든 예측 포즈는 천골 조각 ID 1(SA1) 기준 상대값으로 재표현되며, SA1은 identity로 고정된다.**

즉 절대 위치가 아니라 **상대 정렬만** 평가된다. 주최측은 참가자가 SA1 = identity를 직접
출력하기를 권장한다. 본 구현은 이를 준수한다 (임상 170케이스 전수 검증: 170/170 준수).

메시는 CT 스캐너 원본 좌표계에 있으며 정규화되어 있지 않다.

---

## 2. 평가 지표

**최종 순위 = 5개 지표 각각의 순위를 평균.** 동점 시 평균 TRE가 낮은 쪽.

| 지표 | 정의 | 공간 |
|---|---|---|
| Rotation Error | 조각별 측지 각도 | 도(°) |
| Translation Error | 조각별 **무게중심 이동 거리** | mm |
| TRE | 어셈블리 대응점 거리 | mm |
| **Chamfer Distance** | 어셈블리 Chamfer — **공식 주지표** | 전역 정규화(단위구) |
| Part Accuracy | 조각별 정규화 CD < **0.05** 인 비율 | — |

샘플링: TRE/CD는 뼈 영역당 **5,000점**(면적비례), PA는 조각당 **1,000점**, 시드 고정.

---

## 3. 알고리즘 — 구현된 것

GPU도 학습도 없는 고전 기하 스택. 전부 `reduction.py` 안에 있다.

### 3.1 OBJ 파싱 — `load_fragment_vertices()`

- 스트리밍 파서. `g` / `o` 라인 → 조각 ID (`LI_101`, `fragment 201` 등 다양한 표기 흡수)
- face 토큰으로 정점을 그룹에 귀속. **음수 상대 인덱스** 처리:
  `vi = (len(verts) + vi) if vi < 0 else vi - 1`
- face가 없는 그룹은 선언 순서 기반으로 정점 귀속 (폴백)
- 임상 170케이스 전수 검증: `plan_pl_gt.json` 과 조각 ID **170/170 정확 일치**

### 3.2 Kabsch — `kabsch(P, Q)`

SVD 교차공분산 + **행렬식 부호 보정**으로 반사(reflection)를 배제해 proper rotation 보장.

### 3.3 ICP — `icp()`

point-to-point. `cKDTree.query`로 대응점 → Kabsch → 누적:

```python
R_tot = R @ R_tot
t_tot = R @ t_tot + t
```

최대 30 iter, 20k점 서브샘플(시드 고정), `tol=1e-4`.

### 3.4 Greedy 조립 — `reduce_fragments()`

1. **앵커** = 조각 ID 1(SA1), 없으면 최대 조각 → identity
2. 나머지를 정점 수 내림차순으로 **이미 배치된 조각들의 합집합**에 ICP
3. **수용 가드**: 평균 최근접거리가 개선되고 **and** 회전 ≤ 60° **and** 무게중심 이동 ≤ 80 mm
   일 때만 채택, 아니면 identity
4. 조각별 `except` → 실패 시 identity

> ⚠️ 이 가드는 실제로는 **한 번도 발동하지 않는다**: GT 분포의 p90이 20.4 mm / 21.1°,
> 최댓값이 67.4 mm이므로 60°/80 mm 임계는 사실상 무한대다. 향후 레버를 시도할 땐 이 임계를
> 실측 GT 봉투(≈25°, ≈20 mm)로 조여야 의미가 생긴다.

---

## 4. 왜 identity인가 — 정량적 근거

### 4.1 greedy overlap-ICP = REFUTED

| 방법 | Rotation | Translation |
|---|--:|--:|
| **identity** | **9°** | **46 mm** |
| greedy overlap-ICP | 18° | 66 mm |

**근본 원인**: 정복(reduction)은 **골절면 정합(surface mating)** 이지 **부피 겹침 정합**이 아니다.
ICP를 "이미 배치된 조각들의 합집합"에 걸면 조각을 서로의 **안쪽으로** 끌어당긴다.
겹침을 최소화하는 방향이 아니라 최대화하는 방향으로 움직이는 것이다.

### 4.2 RF pose-regressor = REFUTED (단, 절반은 무효한 반증이었다)

`train_regressor.py` — 앵커 좌표계에서의 특징(무게중심, bbox 크기, 공분산 고유값, 주축,
`log1p(n_verts)`, SA/LI/RI one-hot, `fid % 100`) → `rotvec(3) + translation(3)`.
`RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3)`,
케이스 단위 `GroupKFold(5)`. identity를 이기지 못하면 저장조차 하지 않도록 게이트되어 있고,
실제로 이기지 못해서 `t3_pose_regressor.joblib` 은 존재하지 않는다 (844 샘플: identity 10.6°/133 mm
vs regressor 10.7°/144 mm).

> ⚠️ **[2026-07-21 정정]** 위 133/144 mm 는 **잘못된 지표**로 측정된 값이다. §5 참조.
> translation 축의 반증은 무효이며, 회전 축의 반증만 유효하다.

### 4.3 상수 예측이 최적인가?

해부부위별 최적 상수 포즈를 구해 identity와 비교한 결과 **TRE 0.09 mm 차이**.
즉 **identity가 사실상 최적의 상수 예측기**다. 여기엔 공짜 점심이 없다.

---

## 5. 로컬 GT 게이트

Task 1/2와 달리 **Task 3은 학습셋에 GT가 동봉**되어 있다(`plan_pl_gt.json`).
따라서 개선을 로컬에서 직접 검증할 수 있다 — 이것이 Task 3의 가장 큰 이점이다.

`eval_reduction.py` (본 저장소가 아닌 개발 트리 `code_task3/`에 위치).

### 5.1 🔴 [2026-07-21] 게이트가 translation을 ~13배 틀리게 재고 있었다

```python
# 이전 (틀림)
trans_err = np.linalg.norm(P[:3, 3] - G[:3, 3])     # 변환행렬의 평행이동 "열" 차이

# 수정 (공식 지표)
c = V.mean(axis=0)                                   # 조각 무게중심
trans_err = ‖ (R_p·c + t_p) − (R_g·c + t_g) ‖        # 무게중심의 실제 이동 거리
```

회전은 **원점 기준**으로 적용되는데 이 데이터의 원점은 골반에서 **~930 mm** 떨어져 있다.
따라서 같은 자세 오차라도 lever-arm 때문에 값이 크게 부풀려진다.

| | 잘못된 지표 | 올바른 지표 |
|---|--:|--:|
| identity, 임상 케이스 | ~132 mm | **~10 mm** |

**결과**: `train_regressor.py` 도 같은 잘못된 타깃으로 학습했으므로 **RF 반증의 translation 절반은
무효**다. 재검증 대상이다.

### 5.2 현재 게이트 (5개 공식 지표 전부 측정)

수정 후 게이트는 Rot/Trans뿐 아니라 TRE·Chamfer·Part Accuracy까지 측정한다.
identity 베이스라인 (임상 25케이스, 122 조각):

```
identity | cases  25 frags  122 | Rot  11.01° | Trans  10.41mm | TRE 8.29mm | CD 0.0204 | PA 0.817
```

**어떤 도전자든 이 5개 숫자를 넘어야 채택된다.**

```bash
python eval_reduction.py <clinical_mesh_dir> [--icp] [--n N]
```

---

## 6. 다음 레버

### 6.1 2-조각 레버 (오라클 검증됨, 미구현)

임상 170케이스 오라클 실험: **가장 큰 좌·우 관골 조각 2개만** 정확히 풀면
(표면적의 57.1%, TRE 질량의 78.5%를 차지) 5개 지표가 전부 움직인다.

| 지표 | 개선 |
|---|--:|
| Chamfer Distance | **−47%** |
| TRE | **−39%** |
| Part Accuracy | **+24%** |
| Translation | −32% |
| Rotation | −24% |

**접근**: 미사용 자산인 `template.nii.gz`(480 골반 모델)로 **템플릿 정합**.
**안전장치**: α-감쇠 블렌드. 현재 `reduction.py` 의 수용 가드는 하드 0/1 게이트인데,
α로 연속화하면 **α=0이 identity를 정확히 복원**하므로 하방 위험이 0이다.
가드 임계도 실측 GT 봉투(≈25°, ≈20 mm)로 조인다.

### 6.2 하지 말아야 할 것

- **겹침 기반 ICP 재시도** — 방향이 근본적으로 틀렸다(§4.1)
- **전역 pose 회귀** — 상수 예측기가 이미 identity와 같다(§4.3)
- 진짜 해법은 **골절면 매칭** 또는 joint transformer (FracFormer 계열)

---

## 7. 재현 / 빌드 / 검증

### 7.1 저장소 구조

```
.
├── Dockerfile              python:3.10-slim, GPU 불필요
├── requirements.txt        numpy, scipy 뿐
├── inference/
│   ├── inference.py        ★ 진입점 (기본 identity, ICP는 env-gated)
│   └── reduction.py        OBJ 파싱 + Kabsch + ICP + greedy 조립
└── scripts/build_image.sh
```

### 7.2 로컬 실행

```bash
# 기본 (identity)
python inference/inference.py

# ICP 백엔드 활성 (실험용 — identity보다 나쁘다는 점 유의)
PENGWIN_T3_ICP=1 python inference/inference.py
```

### 7.3 출력 형식

```json
[
  {"fragment_id": 1,   "transformation": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  {"fragment_id": 101, "transformation": [[...]]}
]
```

SA1(=1)은 반드시 identity.

### 7.4 스모크 검증

```python
import json
d = json.load(open("/output/reduction-poses-matrices.json"))
assert all(len(e["transformation"]) == 4 for e in d)
anchor = [e for e in d if e["fragment_id"] == 1]
assert anchor and anchor[0]["transformation"] == [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
print(f"{len(d)} fragments OK")
```

---

## 8. 부록

### 8.1 환경변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `PENGWIN_T3_ICP` | `0` | `1`이면 greedy overlap-ICP 사용 (**권장하지 않음**) |

### 8.2 참고문헌

```bibtex
@article{LIUandYIBULAYIMU2025MEDIA,
  title   = {Preoperative fracture reduction planning for image-guided pelvic trauma
             surgery: A comprehensive pipeline with learning},
  journal = {Medical Image Analysis}, volume = {102}, pages = {103506}, year = {2025},
  doi     = {10.1016/j.media.2025.103506}
}

@article{yibulayimu2025fracformer,
  title   = {FracFormer: Fracture Reduction Planning With Transformer-Based Shape
             Restoration and Fracture Data Simulation},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025}, volume = {44}, number = {8}, pages = {3270-3283},
  doi     = {10.1109/TMI.2025.3561030}
}
```

베이스라인: https://github.com/Sutuk/PENGWIN2026_Task3_Reduction_Baseline

### 8.3 라이선스

MIT — [LICENSE](LICENSE) 참조.
