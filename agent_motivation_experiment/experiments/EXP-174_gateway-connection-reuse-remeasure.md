# EXP-174 — 게이트웨이가 스케줄러 연결을 재사용하게 고친 뒤 8B 함대 재측정 (2026-11-22)

**물음**: EXP-173에서 찾은 측정 경로 결함(게이트웨이의 스케줄러 호출이 임시 포트 고갈로 실패하면 요청을 admission 판정 없이 round-robin으로 엔진에 보내는 것)을 고치면, ① 그 대체 배치가 실제로 0이 되는가, ② 그 상태에서 PolyServe와 FluidServe 후보들의 점수가 얼마인가.

**바꾸는 것은 하나다**: 게이트웨이 바이너리. 함대, trace, 워크로드 설정, 스케줄러 바이너리와 플래그, 드라이버는 EXP-170·172·173과 같다.

## 1. 결함과 수정

**결함(EXP-173에서 로그로 확인)**: 게이트웨이의 스케줄러 client(`pkg/gateway/load-balancer/scheduler_client.go`)는 `&http.Client{Timeout: 2s}`였고 Go 기본 Transport를 썼다. 기본 Transport는 호스트 하나당 유휴 연결을 2개만 남긴다. 스케줄러가 "지금 보낼 인스턴스가 없다"고 답한 요청은 게이트웨이가 붙잡고 있다가 재시도 간격(우리 arm은 500 ms)마다 스케줄러를 다시 부르므로, 붙잡힌 요청이 수백~수천 건이면 호출 대부분이 새 TCP 연결을 열고 응답을 받은 뒤 닫는다. 닫힌 연결은 TIME_WAIT 상태로 약 60초 동안 임시 포트를 잡고 있고, 임시 포트 약 2.8만 개가 다 차면 `connect: cannot assign requested address`가 난다. client는 이 오류를 "스케줄러가 준비되지 않았다"(`ErrorSchedulerNotReady`)로 바꿔 반환하고, `composite_balancer.go`의 `getWithFallback`이 그 요청을 **게이트웨이 자체의 round-robin으로 엔진에 보낸다.** 그 요청은 스케줄러의 admission·routing 판정을 받지 않았는데 채점에서는 "받아들인 요청"으로 세어진다.

**수정(바깥 저장소 커밋 `02ee187`)**:

| 바꾼 것 | 내용 | 이유 |
|---|---|---|
| 스케줄러 client의 Transport | `newSchedulerHTTPClient()`: 기본 Transport를 복제해 호스트당 유휴 연결 1,000개, 유휴 제한 시간 5분. 호출 제한 시간 2초는 그대로 | 원인이 "연결을 매번 새로 여는 것"이므로 연결을 재사용하게 한다. 값은 같은 게이트웨이가 엔진으로 요청을 넘길 때 쓰는 client(`pkg/gateway/forwarder/utils.go`)와 같다 — 새 조정 값을 만들지 않았다 |
| 대체 배치 횟수 counter | `gateway_scheduling_fallback_total`: `getWithFallback`이 대체 배치를 할 때 1 증가. 기동 때부터 0으로 노출 | 게이트웨이 로그는 run마다 보관되지 않아서, 지금까지는 대체 배치가 있었는지 run 기록만으로 알 수 없었다. run 수집기(`llumnix_metrics.py`의 `GATEWAY_METRICS`)에 이름을 넣어 `server_metrics/gateway.jsonl`에 남긴다 |
| 대체 배치 동작 자체 | **바꾸지 않았다** | upstream 동작이고(스케줄러가 죽었을 때 서비스를 유지하는 장치), 기준선 arm의 게이트웨이를 upstream과 다르게 만들지 않기 위해서다. 대신 counter로 run마다 0인지 확인한다 |

**단위 시험**(`scheduler_client_test.go`): 동시 호출 200개 × 20회(4,000 호출)에서 연 TCP 연결 수가 새 client 186개, 기본 Transport 2,832개. `go test ./pkg/gateway/...` 통과.

**바이너리**: `/home/nxclab/tools/staging/gateway-connreuse2`, md5 `3590c1af68f6c2682676f45b0b2428d7`. 지금까지 8B run 전부가 쓴 게이트웨이는 `gateway-notifyfix2`(`05745270`, 2026-11-04 빌드)이고, 그 뒤 게이트웨이가 의존하는 코드의 변경은 `cmd/config/config.go`의 플래그 추가(스케줄러용)뿐이다.

## 2. 구성

EXP-170과 같다: Llama-3.1-8B × 8대(TP=1), `--max-num-seqs 4096`, `--api-server-count 4`, 게이트웨이 `-v 0`·CPU 24·GOMAXPROCS 24, trace `dyn60_shift_m2Am1B_b1045_x620.csv`(논문 trace의 6.2배, 61분, 채점 대상 도착 약 608k건), 부하 생성 36 프로세스. 드라이버 `k8s/exp07/run_exp172_hour.sh`(변형 `shift62`), 체인 `/home/nxclab/scratch/exp174/run.sh`, 로그 `/home/nxclab/tools/exp174_chain.log`, session prefix `exp174h`.

| 순서 | arm | 무엇인가 | 스케줄러 바이너리 | 반복 |
|---|---|---|---|---|
| 1 | `polyservept75` | PolyServe(논문 구성, 8B 프로파일의 tier 표) | `scheduler-gateutil` `7dc2355c` | r1 |
| 2 | `fsxc3` | EXP-172 후보 3: v0.6에서 클래스별 인스턴스 상한을 끄고, 인스턴스마다 KV 가격(dual ascent로 갱신)을 두어 가격이 높은 인스턴스는 KV를 많이 쓸 요청을 받지 않는다 | `scheduler-exp172-c3` `5d52124e` | r1 |
| 3 | `fsxbc2` | EXP-173 B+C2: KV 유입 속도 밸브(인스턴스에 들어오는 KV 양을 나가는 양에 맞춘다) + 첫 토큰 판정을 prefill 대기열의 유량과 Little의 법칙으로 | `scheduler-exp173-bc2` `956149dc` | r1 |
| 4 | `polyservept75` | | | r2 |
| 5 | `fsxc3` | | | r2 |
| 6 | `fsg10dl` | FluidServe v0.6(지금 기준선) — **게이트웨이 바이너리가 바뀐 것이 v0.6 점수를 움직이지 않는지 보는 대조 1회** | `scheduler-gateutil` `7dc2355c` | r1 |

6번은 "같은 세션 비교를 위해 FluidServe를 다시 돌리지 않는다"는 규칙(2026-11-18)의 예외로 넣었다: 이유가 세션 차이가 아니라 **측정 경로의 바이너리가 바뀐 것**이기 때문이다. 기록된 v0.6 run은 대체 배치가 4건뿐이라 달라질 이유가 없다고 예측하지만, 그 예측을 확인하지 않으면 기록된 v0.6(옛 게이트웨이)과 새 run(새 게이트웨이)을 비교할 수 없다. 사용자가 불필요하다고 하면 대기열에서 뺀다(맨 뒤에 둔 이유).

## 3. 판정 규칙 (실행 전에 쓴다)

**(가) 수정이 동작했는가 — run마다**

- run 끝의 `gateway_scheduling_fallback_total`(체인이 게이트웨이 `/metrics`를 직접 읽어 `/home/nxclab/scratch/exp174/gwmetrics_<반복>_<arm>.txt`에 저장하고, run의 `server_metrics/gateway.jsonl`에도 남는다)이 **받아들인 요청의 0.1 % 이하(약 400건)** 이면 그 run은 유효하다. 기록된 v0.6 run도 4건이 있었으므로(스케줄러 호출 2초 초과 등 포트 고갈이 아닌 원인) 0을 요구하지 않는다.
- 넘으면 그 run은 무효이고, **2,000건을 넘으면 체인이 멈춘다**(수정이 실패한 것이므로 나머지 run을 돌리지 않는다).
- 보조 확인: 게이트웨이 파드 로그의 `cannot assign requested address` 줄 수(0이어야 한다; kubelet이 로그를 회전시키므로 0이 아닐 때의 줄 수는 하한이다), FluidServe arm은 판정 로그에 route 줄이 없는 받아들인 요청 수(`exact_bypass.py`)가 counter와 같은 크기인지.

**(나) 비교 — 지표마다 따로**(모든 도착, admitted, 거절률, token goodput, chat·deepresearch·swe 각각의 모든 도착과 admitted). "다르다" = 평균 차이 > max(문턱, 두 조건 각각의 반복 범위).

| 비교 | 문턱 | 무엇을 답하는가 | 예측 |
|---|---|---|---|
| PolyServe 새 2회 vs EXP-170 PolyServe 2회(모든 도착 72.81 / 74.15, admitted 95.91 / 96.44, 거절률 24.00 / 23.03 %, 대체 배치가 받아들인 요청의 9.5 / 6.7 % 이상) | 세션 간: 모든 도착·거절률·클래스별 모든 도착 4.6, admitted 1, goodput 5 % | 대체 배치가 PolyServe 8B 점수를 얼마나 바꿨는가. 차이 없는 지표는 EXP-170 값을 그대로 쓰고, 다른 지표가 있으면 새 값이 8B PolyServe 기준이 된다 | 방향을 모른다. 대체 배치된 요청이 원래 거절될 요청이었다면 거절률이 오르고 admitted가 오른다 |
| v0.6 새 1회 vs 기록된 v0.6 2회(64.71 / 63.03, admitted 96.48 / 97.03, 거절률 32.76 / 34.90 %) | 같은 세션 간 문턱 | 게이트웨이 변경이 v0.6을 움직이는가 | 모든 지표 차이 없음. 하나라도 다르면 8B의 모든 비교를 새 게이트웨이 run끼리만 한다 |
| 후보 3 새 2회 vs PolyServe 새 2회 (같은 체인) | 한 시간 trace: 모든 도착·거절률·클래스별 모든 도착 4, admitted 1, goodput 5 % | 대체 배치 없이 후보 3이 PolyServe와의 격차(EXP-170에서 모든 도착 −9.6)를 메우는가 | EXP-172의 스케줄러가 배치한 요청만 센 값은 모든 도착 69.2 / 80.9 / 70.9, admitted 94.5 / 97.3 / 97.9. 반복 범위가 4점 안으로 줄면 EXP-172의 큰 반복 간 편차(11.7점)는 대체 배치 때문이었던 것 |
| 후보 3 새 2회 vs 기록된 v0.6 2회 | 세션 간 문턱 | 후보 3이 v0.6보다 나은가 | 모든 도착이 5점 이상 높다 |
| B+C2 새 1회 | 1회이므로 "다르다" 판정을 하지 않고 반증으로만 읽는다 | EXP-173에서 스케줄러가 배치한 요청만 센 값(모든 도착 46.15, admitted 99.44)이 대체 배치 없는 실제 측정에서도 나오는가 | 모든 도착 46~59 사이, admitted 98 이상. 모든 도착이 v0.6(63.87)보다 4.6점 넘게 낮으면 "KV 유입 속도 밸브는 받은 요청은 지키지만 받는 양이 줄어든다"가 확정된다. admitted가 96 아래면 첫 토큰 예측이 맞았다는 EXP-173의 결과가 대체 배치에 기댄 것이었으므로 그 결론을 거둔다 |

**주의**: preemption 횟수는 세션을 넘어 비교하지 않는다(같은 체인 안에서만).

## 결과

(run이 끝나는 대로 적는다.)
