"""Sub-agents: researcher (Sonnet), writer (Sonnet), critic (Opus).

Each sub-agent is a thin wrapper around call_with_tools_loop (for writer
and researcher) or generate_structured (for critic). They receive a
system prompt + task-specific context, run their own tool-use loop, and
return structured results to the orchestrator.
"""
