# To run seperate test cases..
# python run_swebench_batch.py --start-index 0 --end-index 10

import time
import tiktoken
from typing import TypedDict, Annotated, Optional, Any
import operator

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    task_id: str
    problem_statement: str
    repo: str
    nonce: str
    plan: str
    code: str
    debug_result: str
    iteration: int
    max_iterations: int
    history: Annotated[list, operator.add]

    # 메트릭 추적용
    metrics_tracker: Optional[object]
    agent_logger: Optional[object]  # Agent 입출력 로깅용

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
        timeout=120,
        top_p=1.0,
        seed=seed,  # 재현성 seed (서버 쪽 --seed 42도 켜셨으니 함께 고정)
    )


def is_pass_verdict(text: str) -> bool:
    normalized = (text or "").strip().upper()
    return normalized.startswith("PASS:")


def is_fail_verdict(text: str) -> bool:
    normalized = (text or "").strip().upper()
    return normalized.startswith("FAIL:")


# 토큰 카운터 (근사치 계산용)
try:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Llama용 근사
except Exception:
    tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 근사 계산"""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


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

    prompt_text = messages[0]["content"] if messages else ""
    input_tokens = count_tokens(prompt_text)

    try:
        response_chunks = []
        for chunk in llm.stream(messages):
            if first_token_time is None:
                first_token_time = time.time()

            if hasattr(chunk, "content") and chunk.content is not None:
                response_chunks.append(chunk.content)

        response_content = "".join(response_chunks)

    except Exception as e:
        # 스트리밍 실패시 일반 호출
        print(f"Warning: Streaming failed, falling back to regular call: {e}")
        response = llm.invoke(messages)
        response_content = response.content
        first_token_time = start_time + (time.time() - start_time) * 0.1

    end_time = time.time()
    latency = end_time - start_time
    output_tokens = count_tokens(response_content)

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
        )

    return response_content


def planning_node(state: AgentState):
    print(f"\n[ITERATION {state['iteration']}] Planning...")

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
    print(f"[ITERATION {state['iteration']}] Coding...")

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
    print(f"[ITERATION {state['iteration']}] Debugging...")

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
        print("✓ PASSED!")
        return END
    if state["iteration"] >= state["max_iterations"]:
        print("✗ Max iterations reached")
        return END

    if state.get("metrics_tracker"):
        state["metrics_tracker"].next_iteration()

    print("→ Retrying with feedback...")
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
        "plan": "",
        "code": "",
        "debug_result": "",
        "iteration": 0,
        "max_iterations": 3,
        "history": [],
        "metrics_tracker": None,
        "agent_logger": None,
        "llm": make_llm(seed=42, temperature=0),  # ✅ demo도 task 전용 llm
    }

    final = agent.invoke(init_state)
    print("\nFinal verdict:", final.get("debug_result"))
