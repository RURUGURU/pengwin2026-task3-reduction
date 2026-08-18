# PENGWIN 2026 Task 3 — v4.5 최종 보존 소스

OBJ 골절 조각마다 4×4 강체변환을 출력하는 AssemblyTransformer 컨테이너다. 이 작업트리는 원본
release `v4.5@e8af95c`를 기반으로 대회 종료 뒤 문서와 미사용 LoRA 파일을 정리한 로컬 archive
branch다. 외부 저장소에는 push하지 않았다.

## 최종 실행 계약

- 입력: 하나의 OBJ 안에 있는 SA/LI/RI fragment group
- 모델: AssemblyTransformer, embed 384, 12 layers, 8 heads, 35,830,304 parameters
- 학습: wide-augmentation simulation pretraining epoch 139; clinical LoRA 없음
- sampling: fragment당 5,000 surface points, seed 0
- anchor: 숫자가 가장 작은 Sacrum fragment를 identity로 정규화
- gate: 중심 이동 30 mm 초과 또는 회전 25도 초과면 identity, 중심 이동 12 mm 미만이면서
  회전 6도 미만이어도 identity
- 실패 처리: 모든 fragment에 올바른 4×4 identity JSON을 쓰지만 성능 하한을 보장하지 않음

모델 archive는 `../model_bundles/v4_5/model.tar.gz`, outer SHA-256
`e392f5d7b412835c7b8a87c4e59cc50ef882b68f173583be3598ad83c0d112a3`, inner checkpoint SHA-256
`2ebd94e1d17322d8636eb6caf89e84ffcdd27f1633ffc562e27fdb54887d4faf`다.

GC snapshot의 `ruruguru v4.5` Final 행은 10/19, MP 10.0이다. 이는 account/submission 행 순위이며
공식 중복 제거 team rank가 아니다. GC model object와 local tar 사이의 byte binding도 제공되지
않았다.

- `inference/`: Grand Challenge wrapper와 형식 fallback
- `baseline/`: v4.5 full-model inference에 필요한 AssemblyTransformer 구현과 설정
- `Dockerfile`, `requirements.txt`: CPU/non-root 컨테이너 계약
- `scripts/build_image.sh`: 로컬 build helper

container rebuild와 전체 checkpoint inference는 이번 archive 정리에서 수행하지 않았다.
