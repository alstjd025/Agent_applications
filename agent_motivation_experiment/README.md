# agent_motivation_experiment — FluidServe 실험의 부하 생성·드라이버·채점

이 저장소(`alstjd025/Agent_applications`)는 FluidServe 프로젝트의 **측정 쪽**이다. 정책 코드와 지금 상태 문서는
바깥 저장소 `llumnix_reproduce`에 있다 — **처음이면 `../../project/README.md`부터 읽는다.**
작업 규칙은 바깥 저장소의 `CLAUDE.md`(이 디렉토리에서 작업해도 상위 디렉토리의 것이 로드된다).

2026-10-11에 정리했다. 2026-05~07의 SGLang·Halo 실험 코드와 문서, 실험 기록 EXP-00~140, 논문에 실리지 않은
그림·드라이버·분석 스크립트는 바깥 저장소의 `archive/Agent_applications/`로 옮겼다(목록과 md5는
`archive/MOVED_2026-10-11_Agent_applications.tsv`).

| 경로 | 무엇 | 고쳐도 되나 |
|---|---|---|
| `run_experiment.py`, `llumnix_deploy.py`, `llumnix_metrics.py`, `metrics_tracker.py`, `agent_logger.py`, `arrival_trace.py` | 러너: 부하 생성, 조건마다 Llumnix cold restart, Prometheus 메트릭 수집 | 측정 경로. **실험이 도는 동안 금지** |
| `workloads/` | 워크로드 정의(chat=ShareGPT, deepresearch=Search Arena, swe=SWE-bench agent 기록 재생, 그리고 셋을 섞는 `mixed_request_level_poisson`) | 측정 경로 |
| `workload_configs/` | 논문 실험이 쓰는 워크로드 설정 43개 | **측정 입력 — 고치지 않는다.** 바꾸려면 새 파일 |
| `traces/` | 도착 trace. 한 시간 trace `dynamic/canonical/dyn60_shift_m2Am1B_b1045.csv` 등 | 측정 입력 |
| `k8s/exp07/` | 논문 실험의 드라이버 18개(`run_exp108_t75.sh` 정적 sweep, `run_exp109_hour.sh` 한 시간 trace, …)와 러너 Job 템플릿 8개 | 측정 경로 |
| `k8s/runner-rbac.yaml`, `runner-job.yaml` | 러너 파드 권한과 일반 Job | |
| `analysis_scripts/` | 채점(`request_level/deadline_ladder_attainment.py` = 논문 규칙 `ladder95`, `exp22_fluidserve.py`의 `load_run`)과 그림이 쓰는 분석 코드 17개 | 고쳐도 측정에 영향 없음 |
| `paper_figures/` | 논문 그림 12개, 스크립트, 그린 값 CSV. 그 디렉토리의 `README.md` | |
| `experiments/` | 새 실험 기록(EXP-141부터) | |
| `results/` | run 결과. 대부분 이 서버에 없다(`../../project/reproduce.md` §1) | **지우지 않는다** |

실행·채점 명령은 `../../project/reproduce.md` §5.
