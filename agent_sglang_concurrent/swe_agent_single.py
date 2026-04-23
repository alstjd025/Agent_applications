# To run seperate test cases..
# python run_swebench_batch.py --start-index 0 --end-index 10

import time
import os
import tiktoken
from typing import TypedDict, Annotated, Optional, Any
import operator

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from metrics_tracker import summarize_tbt_ms
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class AgentState(TypedDict):
    task_id: str
    problem_statement: str
    repo: str
    nonce: str
    log_level: str
    plan: str
    code: str
    debug_result: str
    iteration: int
    max_iterations: int
    history: Annotated[list, operator.add]

    # 메트릭 추적용
    metrics_tracker: Optional[object]
    agent_logger: Optional[object]  # Agent 입출력 로깅용
    console_write: Optional[Any]

    # ✅ task 전용 LLM 인스턴스 (멀티스레딩에서 shared_llm 공유 금지)
    llm: Optional[Any]


# === 단일 SGlang 서버/단일 모델 ===
BASE_URL = "http://localhost:30000/v1"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# ✅ task별로 새 LLM 인스턴스를 만들기 위한 팩토리
def make_llm(seed: int = 42, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key="dummy",
        model=MODEL_ID,
        temperature=temperature,
        # timeout=120,
        timeout=None,
        top_p=1.0,
        seed=seed,  # 재현성 seed (서버 쪽 --seed 42도 켜셨으니 함께 고정)
    )


def is_pass_verdict(text: str) -> bool:
    normalized = (text or "").strip().upper()
    return normalized.startswith("PASS:")


def is_fail_verdict(text: str) -> bool:
    normalized = (text or "").strip().upper()
    return normalized.startswith("FAIL:")


# 토큰 카운터
tokenizer = None
tokenizer_mode = "unknown"
TOKENIZER_SOURCE = os.getenv("LLAMA_TOKENIZER_PATH", MODEL_ID)

if AutoTokenizer is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_SOURCE,
            use_fast=True,
            local_files_only=True,
        )
        tokenizer_mode = f"transformers:{TOKENIZER_SOURCE}"
    except Exception:
        tokenizer = None

if tokenizer is None:
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # fallback 근사
        tokenizer_mode = "tiktoken:gpt-3.5-turbo"
    except Exception:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokenizer_mode = "tiktoken:cl100k_base"


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 계산. 가능하면 Llama tokenizer를 우선 사용."""
    try:
        if tokenizer_mode.startswith("tiktoken:"):
            return len(tokenizer.encode(text))
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return int(len(text.split()) * 1.3)


LOG_LEVEL_ORDER = {"quiet": 0, "info": 1, "debug": 2}


def should_log(state: AgentState, level: str) -> bool:
    current_level = state.get("log_level", "quiet")
    return LOG_LEVEL_ORDER.get(current_level, 0) >= LOG_LEVEL_ORDER.get(level, 0)


def emit_log(state: AgentState, message: str, level: str = "debug") -> None:
    if not should_log(state, level):
        return

    writer = state.get("console_write")
    if callable(writer):
        writer(message)
    else:
        print(message)


def invoke_with_tracking(messages, agent_name: str, state: AgentState):
    """
    LLM 호출 + 메트릭 추적 + 입출력 로깅

    - ✅ shared_llm을 전역으로 공유하지 않고 state['llm']을 사용합니다.
    """
    # ✅ task별 llm 사용 (없으면 생성해서 state에 박아둠)
    llm = state.get("llm")
    if llm is None:
        llm = make_llm(seed=42, temperature=0.7)
        state["llm"] = llm

    start_time = time.time()
    first_token_time = None
    last_stream_chunk_time = None

    prompt_text = messages[0]["content"] if messages else ""
    input_tokens = count_tokens(prompt_text)
    stream_fallback_used = False
    response_chunks = []
    chunk_events = []
    tbt_values_ms = []
    streamed_output_tokens_est = 0
    first_chunk_tokens_est = 0
    content_chunk_idx = 0

    try:
        for chunk in llm.stream(messages):
            chunk_arrival_time = time.time()

            if hasattr(chunk, "content") and chunk.content is not None:
                chunk_text = chunk.content
                if chunk_text:
                    if first_token_time is None:
                        first_token_time = chunk_arrival_time
                    response_chunks.append(chunk_text)
                    chunk_tokens_est = max(count_tokens(chunk_text), 1)
                    streamed_output_tokens_est += chunk_tokens_est

                    if last_stream_chunk_time is None:
                        first_chunk_tokens_est = chunk_tokens_est
                        inter_arrival_ms = None
                    else:
                        inter_arrival_ms = (chunk_arrival_time - last_stream_chunk_time) * 1000.0
                        per_token_tbt_ms = inter_arrival_ms / chunk_tokens_est
                        tbt_values_ms.extend([per_token_tbt_ms] * chunk_tokens_est)

                    chunk_events.append(
                        {
                            "chunk_idx": content_chunk_idx,
                            "arrival_offset_ms": round((chunk_arrival_time - start_time) * 1000.0, 4),
                            "delta_chars": len(chunk_text),
                            "delta_tokens_est": chunk_tokens_est,
                            "inter_arrival_ms": round(inter_arrival_ms, 4) if inter_arrival_ms is not None else None,
                        }
                    )
                    last_stream_chunk_time = chunk_arrival_time
                    content_chunk_idx += 1

        response_content = "".join(response_chunks)

    except Exception as e:
        # 스트리밍 실패시 일반 호출
        stream_fallback_used = True
        emit_log(
            state,
            f"Warning: Streaming failed, falling back to regular call: {e}",
            level="info",
        )
        response = llm.invoke(messages)
        response_content = response.content
        first_token_time = start_time + (time.time() - start_time) * 0.1

    end_time = time.time()
    latency = end_time - start_time
    output_tokens = count_tokens(response_content)
    tbt_summary = summarize_tbt_ms(tbt_values_ms)
    tbt_summary.update(
        {
            "stream_chunks": len(chunk_events),
            "streamed_output_tokens_est": streamed_output_tokens_est,
            "first_chunk_tokens_est": first_chunk_tokens_est,
        }
    )
    tbt_detail = {
        "nonce": state["nonce"],
        "start_time": start_time,
        "end_time": end_time,
        "latency_s": round(latency, 6),
        "first_token_latency_ms": round((first_token_time - start_time) * 1000.0, 4) if first_token_time is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "streamed_output_tokens_est": streamed_output_tokens_est,
        "first_chunk_tokens_est": first_chunk_tokens_est,
        "chunk_count": len(chunk_events),
        "chunk_events": chunk_events,
        "tbt_summary": {
            "available": tbt_summary.get("available"),
            "mean_ms": round(tbt_summary["mean_ms"], 4) if tbt_summary.get("mean_ms") is not None else None,
            "p50_ms": round(tbt_summary["p50_ms"], 4) if tbt_summary.get("p50_ms") is not None else None,
            "p75_ms": round(tbt_summary["p75_ms"], 4) if tbt_summary.get("p75_ms") is not None else None,
            "p80_ms": round(tbt_summary["p80_ms"], 4) if tbt_summary.get("p80_ms") is not None else None,
            "p85_ms": round(tbt_summary["p85_ms"], 4) if tbt_summary.get("p85_ms") is not None else None,
            "p90_ms": round(tbt_summary["p90_ms"], 4) if tbt_summary.get("p90_ms") is not None else None,
            "p95_ms": round(tbt_summary["p95_ms"], 4) if tbt_summary.get("p95_ms") is not None else None,
            "max_ms": round(tbt_summary["max_ms"], 4) if tbt_summary.get("max_ms") is not None else None,
            "sample_count": tbt_summary.get("sample_count"),
        },
    }

    # Agent 입출력 로깅
    if state.get("agent_logger"):
        state["agent_logger"].log_agent_call(
            agent_name=agent_name,
            input_prompt=prompt_text,
            output_response=response_content,
            latency=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # 메트릭 기록
    if state.get("metrics_tracker"):
        success = None
        if agent_name == "debugging":
            success = is_pass_verdict(response_content)

        state["metrics_tracker"].record_agent_call(
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            first_token_time=first_token_time,
            success=success,
            tokenizer_mode=tokenizer_mode,
            stream_fallback_used=stream_fallback_used,
            tbt_summary=tbt_summary,
            tbt_detail=tbt_detail,
        )

    return response_content


def planning_node(state: AgentState):
    emit_log(state, f"[ITERATION {state['iteration']}] Planning...", level="debug")

    # Iteration 로깅 시작
    if state.get("agent_logger"):
        state["agent_logger"].start_iteration(state["iteration"])

    feedback = ""
    if state.get("debug_result") and is_fail_verdict(state["debug_result"]):
        feedback = f"\n\nPrevious attempt failed:\n{state['debug_result']}\n\nPlease revise your plan."

    prompt = f"""Experiment metadata: nonce={state['nonce']}

You are a senior software engineer analyzing a bug report.

Bug Report:
{state['problem_statement']}

Repository: {state['repo']}
{feedback}

Create a detailed plan to fix this bug:
1. Which files need to be modified?
2. What functions/classes are involved?
3. What is the root cause?
4. How should we fix it?

Plan:"""

    messages = [{"role": "user", "content": prompt}]
    response_content = invoke_with_tracking(messages, "planning", state)

    return {
        "plan": response_content,
        "history": [{"role": "planner", "iteration": state["iteration"], "content": response_content}],
    }


def coding_node(state: AgentState):
    emit_log(state, f"[ITERATION {state['iteration']}] Coding...", level="debug")

    prompt = f"""Experiment metadata: nonce={state['nonce']}

You are an expert programmer. Implement the following fix.

Plan:
{state['plan']}

Bug Report:
{state['problem_statement']}

Generate the complete fixed code or a patch in unified diff format.

Code:"""

    messages = [{"role": "user", "content": prompt}]
    response_content = invoke_with_tracking(messages, "coding", state)

    return {
        "code": response_content,
        "history": [{"role": "coder", "iteration": state["iteration"], "content": response_content}],
    }


def debugging_node(state: AgentState):
    emit_log(state, f"[ITERATION {state['iteration']}] Debugging...", level="debug")

    prompt = f"""Experiment metadata: nonce={state['nonce']}

You are a senior code reviewer. Verify if this code correctly fixes the bug.

Original Bug:
{state['problem_statement']}

Plan:
{state['plan']}

Generated Code:
{state['code']}

Evaluate whether the generated code would correctly fix the bug.

Respond with exactly one of these formats:
- "PASS: <brief explanation of why the code should work and what part of the bug it addresses>"
- "FAIL: <brief explanation of what is wrong, what is missing, or why it would not work>"

If you respond with FAIL, the explanation should be useful for the next planning step.

Verdict:"""

    messages = [{"role": "user", "content": prompt}]
    response_content = invoke_with_tracking(messages, "debugging", state)

    return {
        "debug_result": response_content,
        "iteration": state["iteration"] + 1,
        "history": [{"role": "debugger", "iteration": state["iteration"], "content": response_content}],
    }


def should_continue(state: AgentState):
    if is_pass_verdict(state["debug_result"]):
        emit_log(state, "✓ PASSED!", level="debug")
        return END
    if state["iteration"] >= state["max_iterations"]:
        emit_log(state, "✗ Max iterations reached", level="debug")
        return END

    if state.get("metrics_tracker"):
        state["metrics_tracker"].next_iteration()

    emit_log(state, "→ Retrying with feedback...", level="debug")
    return "planning"


# LangGraph 워크플로우 구성
workflow = StateGraph(AgentState)
workflow.add_node("planning", planning_node)
workflow.add_node("coding", coding_node)
workflow.add_node("debugging", debugging_node)

workflow.set_entry_point("planning")
workflow.add_edge("planning", "coding")
workflow.add_edge("coding", "debugging")
workflow.add_conditional_edges("debugging", should_continue, {"planning": "planning", END: END})

agent = workflow.compile()


if __name__ == "__main__":
    print("Agent (single model, per-task llm, seed=42) initialized successfully!")

    init_state: AgentState = {
        "task_id": "demo",
        "problem_statement": "There is a bug in function foo() that crashes on empty input.",
        "repo": "demo-repo",
        "nonce": "demo-0-planning",
        "log_level": "debug",
        "plan": "",
        "code": "",
        "debug_result": "",
        "iteration": 0,
        "max_iterations": 3,
        "history": [],
        "metrics_tracker": None,
        "agent_logger": None,
        "console_write": None,
        "llm": make_llm(seed=42, temperature=0),  # ✅ demo도 task 전용 llm
    }

    final = agent.invoke(init_state)
    print("\nFinal verdict:", final.get("debug_result"))
