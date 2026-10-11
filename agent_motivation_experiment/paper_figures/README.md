# paper_figures — 제출한 논문(NSDI '27 submission #515)에 실린 그림만

2026-10-11에 이 디렉토리를 정리했다. **여기 남은 것은 논문에 실린 데이터 그림 12개, 그것을 그린 스크립트,
그 스크립트가 import하는 모듈, 그리고 그린 값을 담은 CSV뿐이다.** 그 밖의 그림 약 400개와 그림 출처의
정본이던 긴 README(3,207줄: 그림마다 run 집합, n, 채점 규칙, 캡션이 적어야 할 것, 말하면 안 되는 것)는
`archive/Agent_applications/agent_motivation_experiment/paper_figures/`에 있다.

그림 하나하나가 무엇을 근거로 무엇을 주장하는지, 같이 적어야 할 한계는 **`project/evidence.md`**에 있다.
이 파일은 "어느 파일이 어느 그림이고 어떻게 다시 그리나"만 적는다.

## 그림 12개

| 논문 | PDF | 스크립트 | 그린 값 (CSV) | 쓰는 실험 |
|---|---|---|---|---|
| Fig 3 | `arrival_and_mix.pdf` | `fig_arrival_and_mix.py` | `arrival_and_mix.csv` | Azure LLM inference trace 원본(`traces/azure/raw/`) |
| Fig 4 | `class_mix_hour_abs_2arms_pl_zoom_all4_batch_att.pdf` | `fig_class_mix_hour.py` ⚠ 옵션 미기록 | 같은 이름 `.csv` | EXP-109 (llm-d, PolyServe, 한 시간 trace) |
| Fig 5 | `engine_tbt.pdf` | `fig_engine_tbt.py` | `engine_tbt.csv` | EXP-109 |
| Fig 6 | `length_refinement_weighted.pdf` | `fig_length_refinement.py --weighted` | `length_refinement.csv`, `length_refinement_distribution.csv` | EXP-64, EXP-108 |
| Fig 11 | `hour_token_goodput_3panel.pdf` | `fig_hour_goodput_split.py` | `hour_goodput_split.csv` | EXP-109 (Llama-70B), EXP-113 (Qwen-72B), EXP-125·126 (Llama-8B, 8 인스턴스) |
| Fig 12 | `hour_request_goodput_3panel.pdf` | `fig_hour_goodput_split.py` | `hour_goodput_split.csv` | 위와 같다 |
| Fig 13 | `class_latency_outcome_grid.pdf` | `fig_class_latency_outcome_rows.py` | `class_latency_outcome_rows.csv` | EXP-109 |
| Fig 14 | `motivation_throughput_vs_goodput_4panel_t75_withfs_col.pdf` | `fig_motivation_tg_4panel.py` | `motivation_throughput_vs_goodput_4panel_t75_withfs.csv` | EXP-108, EXP-77 (vLLM router), EXP-110 (llm-d 10 req/s 칸) |
| Fig 15 | `exp131_horizon_goodput.pdf` | `fig_exp131_horizon.py` | `exp131_horizon_goodput.csv` | EXP-131 (Llama-70B), EXP-136 (Qwen-72B) |
| Fig 16 | `kv_batch_attain_4arms.pdf` | `fig_kv_batch_attain.py` | `kv_batch_attain_4arms.csv` | EXP-109, EXP-132, EXP-138 |
| Fig 17 | `class_mix_hour_abs_exp138_batch_tbt.pdf` | `fig_class_mix_hour.py --preset exp138` ⚠ 나머지 옵션 미기록 | 같은 이름 `.csv` | EXP-138 (⚠ 실험 기록 파일이 없다) |
| Fig 18 | `exp139_slo_scale_goodput_lrcol.pdf` | `fig_exp139_slo_scale.py --panels-lr-col` | `exp139_slo_scale_goodput.csv` | EXP-139 |

Figure 2·7~10은 도식이다. **Figure 1(정책별 token goodput 대 admitted SLO 달성률 산점도)과 Table 1(워크로드
길이 통계)은 이 저장소에서 만든 스크립트를 찾지 못했다** — `project/revision.md`의 미해결 항목.

## 다시 그리려면

```bash
cd Agent_applications/agent_motivation_experiment
python3 paper_figures/fig_engine_tbt.py          # 예. 다른 그림도 위 표의 스크립트를 같은 식으로
```

- **run 디렉토리가 이 서버에 없다.** 그림은 `results/<run>/`의 `metrics.csv`·`server_metrics/`와
  `results/aggregate_analysis/` 아래 채점 캐시(`ladder95/`, `ladder95_halved/`, `class_mix/`, `exp131_ablation/`)를
  읽는다. 데이터가 어디 있는지는 `archive/llumnix_reproduce/ms_dev/scripts/clean_backup_scripts/README.md` §1.
- 채점 캐시를 만드는 스크립트: `analysis_scripts/request_level/deadline_ladder_attainment.py`(규칙 `ladder95`),
  `build_paper_ladder_table.py`, `build_class_mix_tables.py`, `exp131_figures.py`, `exp139_slo_scale.py`.
- Fig 4·17의 `--suffix`는 출력 이름(`class_mix_hour_abs{suffix}{_att|_tbt}.pdf`)으로 보아 각각
  `_2arms_pl_zoom_all4_batch`, `_exp138_batch`다. 그 밖의 옵션(`--arms`, `--zoom`, `--height batch` 등)은 기록이
  없으므로, 다시 그릴 때는 CSV와 PDF를 대조해 맞춘다.
- 모든 스크립트는 그린 값을 같은 이름의 CSV로 쓴다. 다시 그린 뒤에는 **새 CSV와 이 디렉토리의 CSV를 대조**한다.

## 여기 남은 보조 모듈

`paper_style.py`(공통 스타일), 그리고 위 스크립트가 import하는 `fig_azure_rate_and_mix.py`, `fig_engine_window.py`,
`fig_load_and_engine.py`, `fig_load_engine_tbt.py`, `fig_two_models_hour.py`, `fig_two_models_hour_exp126.py`.
이 여섯은 그 자체로는 논문 그림을 만들지 않지만 지우면 import가 깨진다.
