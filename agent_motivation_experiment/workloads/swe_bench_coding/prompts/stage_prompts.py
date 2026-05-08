"""Agent stage definitions and stage-specific user prompts."""


class AgentStage:
    """Stage definitions for the simulated coding agent workflow."""
    UNDERSTAND = "understand"   # Analyze the issue
    LOCATE = "locate"           # Search/read files
    PLAN = "plan"               # Design solution
    IMPLEMENT = "implement"     # Write/edit code
    VERIFY = "verify"           # Run tests
    DEBUG = "debug"             # Diagnose test failures


STAGE_PROMPTS = {
    AgentStage.UNDERSTAND: (
        "You have received the issue description above. Analyze it carefully. "
        "Identify the root cause, affected components, and the scope of changes "
        "needed. What files and functions are likely involved?"
    ),
    AgentStage.LOCATE: (
        "Search the codebase for the relevant code. Use the search_files tool to "
        "find files matching the key terms from your analysis. Read the files you "
        "find to understand the current implementation."
    ),
    AgentStage.PLAN: (
        "Based on your understanding of the code and the issue, design a solution. "
        "Explain your approach step by step, including which files to modify and "
        "what changes to make."
    ),
    AgentStage.IMPLEMENT: (
        "Implement the planned changes. Write the exact code modifications needed. "
        "Use the edit_file tool to make targeted changes."
    ),
    AgentStage.VERIFY: (
        "Run the test suite to verify your changes. Use the run_tests tool. "
        "Analyze the results carefully -- do all tests pass? Are there any regressions?"
    ),
    AgentStage.DEBUG: (
        "The tests revealed failures. Analyze the test output to identify what went "
        "wrong. What is the root cause of the failure? How should the implementation "
        "be adjusted?"
    ),
}
