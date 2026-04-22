"""Round 3 Phase 2: Claude-Code-style orchestrator agent for email personalization.

V2 replaces V1's write->judge->refine pipeline per recipient with an Opus
orchestrator that dispatches Sonnet sub-agents (researcher, writer) and an
Opus critic, using a generic tool-use loop modeled on Claude Code's
QueryEngine pattern.

Activated via env var ORCHESTRATOR_V2=1. V1 (pipeline.py's write/judge/refine)
remains the default. The seam lives at the top of
PersonalizerPipeline._personalize_one_recipient in pipeline.py.
"""
