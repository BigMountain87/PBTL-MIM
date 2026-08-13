# MODEL_MANIFEST — 학습 모델 체크포인트 (초안, 2026-08-03)

논문 Data Availability의 "trained models" 이행을 위한 생존 체크포인트 목록.

## 출처 상태(provenance) 정의

- **original**: 논문 결과를 만든 시기·머신의 산출물로 확인됨
- **likely-original**: 결과 npz 생성 직전 타임스탬프 — 정본 유력하나 해시 대조 불가
- **representative**: 동일 스크립트의 재실행분 — 논문 표를 만든 바로 그 파일은 아님
  (TMM 사전훈련의 부동소수점 비결정성으로 실행 간 미세 차이, 성능 영향 ~0.1%p 이내:
  2026-07-22 재훈련 검증에서 확인)

## 소실분 — 경위 (코드·타임스탬프로 확인, 2026-08-03)

구조별 저장 방식이 달라 결과가 갈렸다.

| 구조 | 스크립트의 저장 동작 | 결과 |
|---|---|---|
| A | `pbtl_A_redesign.py:229-230` — 저장하되 **구세대와 동일 파일명**(`pretrained_m0_tmm.pt`, `pretrained_mphys_tmm.pt`) | A 본실험은 맥에서 실행(6/23, 결과 해시로 확정)했으나 해당 이름의 맥 파일은 3월자 → **덮어쓰기/미보존으로 소실**. 리눅스의 동명 파일은 6/29 재실행분(=representative) |
| B | `pbtl_B_redesign.py:162-163` — **전용 파일명**(`_tmm_B.pt`) | **생존**. 6/22 21:54 저장 → B 결과(6/23 04:17) 직전 = 정본 유력 |
| C | `pbtl_C_v2_redesign.py` — **디스크에 저장하지 않음**. 프로세스 내에서 사전훈련 후 `deepcopy`로 바로 미세조정에 투입(:292, :297) | **설계상 미저장** — 덮어쓰기가 아니라 처음부터 파일이 없음 |

**S17 (geometry-constraint sensitivity)**
- B·C: **vast.ai에서 2026-06-27 실행**(`inv3_sens_BC_vast.py`). 결과 npz와 로그만 회수
  (`results/vast_inv3_20260627/`, `logs/vast_inv3_20260627/`) → **모델은 인스턴스 종료와 함께 소멸**
- A: 집 머신에서 실행(`inv3_sens_A_home2.py`)했으나 출력 경로가 `/tmp/inv3_sens_A_result.npz`
  → 결과는 회수·아카이브되었으나(맥 `results/`, 매니페스트 일치) **모델은 미저장**

**공통**: 모든 소실분은 공개된 드라이버 스크립트로 재생성 가능하다. 단 TMM 사전훈련이
비결정적이라 가중치는 실행마다 미세하게 다르며, 성능 영향은 ~0.1%p 이내
(2026-07-22 독립 재훈련 검증: from-scratch 모델은 bit-exact 재현, 전이 모델만 미세 산포).

## 리눅스 생존분 (`~/mim_novel/results`)

| 파일 | 역할 | 상태 | 크기(B) | 수정일 | SHA-256 |
|---|---|---|---|---|---|
| `pretrained_m0_tmm.pt` | A: M0 TMM 사전훈련 (6/29 재실행분 — 6/23 결과 생성 이후) | representative | 2,279,560 | 2026-06-29 23:42 | `12635530a4df30d3…` |
| `pretrained_mphys_tmm.pt` | A: M_phys TMM 사전훈련 (6/29 재실행분) | representative | 2,297,094 | 2026-06-29 23:42 | `6f9692b4c8634f35…` |
| `pretrained_m0_tmm_B.pt` | B: M0 TMM 사전훈련 (6/22 — 6/23 B 결과 직전, 정본 유력) | likely-original | 2,277,596 | 2026-06-22 21:54 | `a6fc72bcd2af8f8e…` |
| `pretrained_mphys_tmm_B.pt` | B: M_phys TMM 사전훈련 (6/22 — 정본 유력) | likely-original | 2,291,098 | 2026-06-22 21:54 | `13c317f05d187a05…` |
| `pretrained_m0_rand_redesign.pt` | 랜덤 대조군 M_rand 사전훈련 (6/24) | original | 2,280,044 | 2026-06-24 08:35 | `3e3f2d63e8a5ec6b…` |
| `pretrained_m0_tmm_redesign.pt` | 10seed 드라이버용 M0 사전훈련 (6/24) | original | 2,280,002 | 2026-06-24 08:34 | `a0a8dcdeafa5d946…` |
| `pretrained_m0_au_tmm_jc.pt` | Au(J&C) 교차재료: M0 (6/29) | original | 2,277,828 | 2026-06-29 14:26 | `2cb8f09f192f94e8…` |
| `pretrained_mphys_au_tmm_jc.pt` | Au(J&C) 교차재료: M_phys (6/29) | original | 2,291,266 | 2026-06-29 14:26 | `3b8140d9309b2f8c…` |
| `mfnn_lf_redesign.pt` | deep composite MF-NN 저충실도(LF) 부분 (6/24) | original | 2,411,550 | 2026-06-24 14:50 | `11640d6e9fe43fac…` |
| `fs_lf_5000_redesign.pt` | Full-Spectrum 변형(S8) LF 사전훈련 (6/24) | original | 2,512,412 | 2026-06-24 12:54 | `294f073a54b93233…` |

전체 해시는 `MODEL_HASHES.txt` 참조.

## 추가 생존분 — S13 (역설계 시연)

논문 보충 S13은 "**earlier visible-band (380–780 nm, fixed N=5) surrogate**로 수행했으며 정정
파이프라인에서 재검증하지 않았다"고 명시한다. 해당 모델이 양 머신에 보존되어 있으며,
3월자인 것이 논문 서술과 정합한다.

| 파일 | 역할 | 상태 | 크기(B) | 수정일 | SHA-256 |
|---|---|---|---|---|---|
| `inverse_design/m0_350.pt` | S13 역설계 시연에 사용된 가시광판 surrogate | disclosed-legacy | 2,277,125 | 2026-03-12 02:51 | `43b707ba71b62875…` |
| `inverse_design/mphys_350.pt` | S13 역설계 시연에 사용된 가시광판 surrogate | disclosed-legacy | 2,297,033 | 2026-03-12 02:51 | `721c269ebc618c83…` |

## 탐색 완결성 기록 (2026-08-03)

A 구조 정정판 사전훈련 체크포인트(`pretrained_m0_tmm.pt` / `pretrained_mphys_tmm.pt`의
6월 20–25일 사본)를 다음 범위에서 전수 탐색했으나 **발견되지 않았다**:

- 맥 전체 Spotlight (`mdfind`) — 6월자 `.pt`는 `_redesign.pt` 2건뿐
- 맥 백업·스냅샷 디렉토리: `PINN2_provenance_2026-06-19`(.pt 없음), `PINN(backup)`,
  `Library/CloudStorage/Dropbox/PINN` — 모두 **다른 프로젝트(PINN spectrum, 2월자)**
- 리눅스 `~/mim_novel`, `~/PINN2/mim_novel`, `~/Nextcloud/Research/PINN/mim_novel`
- 맥의 `pretrained_mphys_tmm.pt` 전 사본: 3월 15일자 2건뿐 (가시광 구판)

→ **A 구조 사전훈련 체크포인트는 영구 소실로 확정.** 재생성 경로는 위 "공통" 항목 참조.

## 맥 사본 (`~/PINN2/mim_novel/results/`, 2026-08-03 Nextcloud 경유 수집)

Structure A 본실험 결과(`pbtl_A_redesign_10seed.npz`, SHA-256 `c1e65a73…`)가 **맥에서 생성**되었음이
매니페스트 해시 대조로 확정되었으므로, 맥의 6월자 체크포인트도 함께 보존한다.
같은 파일명의 리눅스본과 **해시가 다르다** — TMM 사전훈련의 부동소수점 비결정성 때문이며,
두 머신에서 각각 사전훈련이 실행된 결과다. 어느 쪽이 논문 표를 만든 파일인지는 확정 불가.

| 파일 | 머신 | 크기(B) | 수정일 | SHA-256 |
|---|---|---|---|---|
| `pretrained_m0_rand_redesign.pt` | **맥** | 2,279,969 | 2026-06-24 09:21 | `4e8f2e049ff3514c…` |
| `pretrained_m0_rand_redesign.pt` | 리눅스 | 2,280,044 | 2026-06-24 08:35 | `3e3f2d63e8a5…` |
| `pretrained_m0_tmm_redesign.pt` | **맥** | 2,279,925 | 2026-06-24 08:49 | `a6ed0e491b52dcbc…` |
| `pretrained_m0_tmm_redesign.pt` | 리눅스 | 2,280,002 | 2026-06-24 08:34 | `a0a8dcdeafa5…` |

릴리스 시 맥본을 정본으로 두고 파일명에 `_mac`/`_linux` 접미사를 붙여 둘 다 보존하거나,
맥본만 공개하고 본 문서에 리눅스본 해시를 병기하는 방식 중 택일.
## 제외 (구세대, 릴리스 비대상)

- 3월자 가시광 파이프라인 체크포인트(`pretrained_*_tmm.pt` 3월판, `_C.pt`, `_au_tmm.pt`)
- `pt_*.pt` 5종 (아키텍처 탐색 실험용)
