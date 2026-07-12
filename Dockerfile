# PENGWIN 2026 Task 3 (PENGWIN-Reduction) — 경량 ICP 조립 컨테이너.
#
# Task 1/2 와 달리 이 태스크는 **분할이 아니라 3D 강체 정합/조립**이다. 딥러닝/GPU/nnUNet 이
# 전혀 필요 없으므로 무거운 pytorch/cuda 베이스 대신 **python:3.10-slim** 을 쓴다(이미지 수백MB).
#
# Layout:
#   /opt/app/inference/inference.py   -> ENTRYPOINT (OBJ 읽기 → reduce → JSON)
#   /opt/app/inference/reduction.py   -> 고전 ICP greedy 조립 백엔드(numpy/scipy)
#
# Grand Challenge I/O (--network none):
#   /input/peripelvic-fracture-fragments-meshes.obj   (읽기전용)
#   /output/reduction-poses-matrices.json             (쓰기)
#   /opt/ml/model/                                     (모델 tarball, 이 베이스라인은 미사용)

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/app

# --- Python deps (순수 CPU; numpy/scipy만) -----------------------------------
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /opt/app/requirements.txt

# --- App code (self-contained; code_task3 COPY 불필요) -----------------------
COPY inference /opt/app/inference

# --- Runtime environment ---------------------------------------------------
# 비root user 는 /home/user 쓰기권한이 없어 일부 라이브러리 캐시가 PermissionError 를 낸다.
# HOME/캐시를 GC 가 허용하는 /tmp 로 돌린다(Task 1 컨테이너와 동일 규약).
ENV PYTHONPATH=/opt/app:/opt/app/inference \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

# Grand Challenge 보안정책: 컨테이너는 root 로 실행하면 안 된다.
RUN groupadd -r user && useradd --no-log-init -r -g user user
USER user:user

# GC 는 --network none, 추가 인자 없이 실행.
ENTRYPOINT ["python", "/opt/app/inference/inference.py"]
