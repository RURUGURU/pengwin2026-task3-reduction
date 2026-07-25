#!/bin/bash
# PENGWIN 2026 Task 3 (Reduction) v4.0 학습형 AssemblyTransformer 컨테이너 빌드
# (torch CPU + trimesh + lightning + hydra; 실패 시 identity 폴백).
#
# 빌드 컨텍스트 = 저장소 루트(이 스크립트의 한 단계 위). Dockerfile 이 inference/ 와
# requirements.txt 를 COPY 할 수 있도록 한다.
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-pengwin-task3-reduction:latest}"

cd "$(dirname "$0")/.."
docker build \
    -t "$IMAGE_TAG" \
    .

echo
echo "Built image: $IMAGE_TAG"
docker images "$IMAGE_TAG" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedAt}}"
