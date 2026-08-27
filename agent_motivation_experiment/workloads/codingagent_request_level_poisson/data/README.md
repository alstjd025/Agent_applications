# SWE transcripts — 워크로드의 입력이지 실험 결과가 아니다

여기 있는 세 파일은 swe 클래스가 그대로 replay하는 프롬프트다. **없으면 이 워크로드를
쓰는 실험이 하나도 안 돈다.**

| 파일 | 크기 | 무엇 |
|---|---|---|
| `transcript_swe_calls.jsonl` | 1.5 GB | 원본 녹화 전체. EXP-10 시절 템플릿만 참조한다 |
| `transcript_swe_calls_mix1500.jsonl` | 170 MB | 그중 1,500 레코드. 아래 짧은 판의 입력 |
| `transcript_swe_short7k_mix1500.jsonl` | 52 MB | **지금 모든 워크로드 설정이 가리키는 파일.** 입력 평균 6,804 토큰 |

## 왜 `results/` 밖에 있나 (2026-08-27에 옮겼다)

**전에는 `results/exp10_transcript/`에 있었고, 그래서 지워졌다.** 결과 디렉토리를 정리하는
작업이 워크로드의 입력을 같이 지울 수 있는 자리였다. 결과는 다시 만들면 되지만 이것은
**입력**이라 없으면 아무것도 재현할 수 없다.

## 복구 절차 (또 없어지면)

1. 백업 `/NHNHOME/NXC_ROOT/NXC13/Agent_applications/agent_motivation_experiment/results/exp10_transcript/`
   에서 `transcript_swe_calls.jsonl`과 `transcript_swe_calls_mix1500.jsonl`을 가져온다.
2. 짧은 판은 **재생성한다** — `build_short_transcript.py`에 난수·시드·시각이 없어
   같은 입력이면 같은 출력이다:

```bash
python workloads/codingagent_request_level_poisson/build_short_transcript.py \
  --in  workloads/codingagent_request_level_poisson/data/transcript_swe_calls_mix1500.jsonl \
  --out workloads/codingagent_request_level_poisson/data/transcript_swe_short7k_mix1500.jsonl \
  --target-mean-tokens 7000
```

3. **재생성이 맞는지 확인한다.** 스크립트가 찍는 값이 아래와 같아야 한다. 이 값들은
   `workloads/mixed_request_level_poisson/workload.py`, `traces/dynamic/build_dynamic_mix_trace.py`,
   `analysis_scripts/request_level/polyserve_allocation_model.py` 세 곳에 독립적으로 적혀 있다.

```
mean input tokens        22474 -> 6812
structural prefix share  81.5% -> 78.9%
1500 줄
```

4. **더 강한 확인**: 이미 돌아간 run과 요청 단위로 대조한다. run의 `metrics.csv`에서
   swe의 `task_id`에서 `__r\d+$`를 떼면 transcript의 `request_id`이고, `input_tokens`가
   `recorded_input_tokens`와 정확히 같아야 한다. 2026-08-27 복구 때 EXP-104 두 run의
   12,886건에서 **없는 레코드 0개, 입력 토큰 평균 차이 +0.0**으로 확인했다.
