# AGENTS.md

## Scope
This file tells codeagent how to work in this repo.
Keep it short, stable, and execution-oriented.
Do not restate long project summaries or stage plans here.

## Canonical path
Default experiment path:

`XHandHoraScrewDriver -> PPO teacher -> current student -> evaluation/export`

Current student is **not** a second-stage RL policy.
It is an imitation-distillation student built from:
- action BC
- latent distillation
- adapter-based adaptation

## What codeagent should do
Codeagent should focus on local execution work:

- read only the files needed for the current task
- preserve the current teacher-student pipeline unless the task explicitly requires otherwise
- prefer small, reversible changes over broad refactors
- keep existing train/eval/export entrypoints usable
- treat Hora as the default active path unless told otherwise
- use `PLANS_v2.md` for current priorities and milestones, not this file

## What codeagent can handle directly
Codeagent may directly do:

- local bug fixes
- config wiring
- training/eval/export script fixes
- student-side implementation changes inside the current pipeline
- baseline reproduction
- small ablations and logging improvements
- documentation updates tied to real code changes

No GPT escalation is needed for these if the work stays inside the current plan.

## When codeagent must escalate
Codeagent must stop and write `codeagent_issue.md` if any of the following happens:

- the canonical path no longer seems valid
- the current student design is misunderstood or conflicts with implementation
- diffusion work requires changing the project goal, milestone, or baseline set
- the change is no longer local and would require major refactor
- teacher-student compatibility breaks in a nontrivial way
- Pasini or external repos become necessary for the current stage
- diffusion cannot show value beyond the current student baseline

## Required issue format
If escalation is needed, `codeagent_issue.md` must contain:

- task background
- current blocker
- evidence
- what has been tried
- local conclusion
- recommended next action

## Governance boundary
`AGENTS.md` is for stable workflow rules only.
`PLANS_v2.md` is for current-stage goals, milestones, priorities, and acceptance criteria.
If something is stage-specific, put it in `PLANS_v2.md`, not here.

## Session Handoff (Required)
After each meaningful execution session, codeagent must update:

- `docs/session_handoff_v2.md` (the only running handoff for Plan v2 execution)

`docs/session_handoff.md` is reserved for Plan v1 historical indexing/archive only, not for ongoing Plan v2 logging.

The handoff update must include:

- what milestone/subgoal was targeted
- what changed (files + behavior impact)
- what was verified (commands + key outcomes)
- what remains blocked/risky
- the single recommended next step

Purpose:

- allow the user to track progress quickly
- let a new session agent onboard fast without replaying full history

## Session Bootstrap (Required)
At the beginning of a new execution session, codeagent must read:

- `docs/session_handoff_v2.md`
- `docs/stage_acceptance_summary.md`

Optional historical context only (when needed):
- `docs/session_handoff.md`

Before any code change or new experiment, codeagent must also do a quick bootstrap check:

- confirm the latest "single recommended next step" from `docs/session_handoff_v2.md`
- avoid rerunning already-failed settings unless testing a clear fix hypothesis
- record the current target milestone/subgoal in the first execution update

Purpose:

- recover the latest experiment state before new changes
- avoid repeating already-failed settings or redundant runs

## Continuous Execution Preference
Unless the user explicitly pauses or redirects, codeagent should run in continuous execution mode:

- continue plan-aligned local validation/probe work across multiple ideas in one stretch
- do not stop for per-probe confirmation when the work is routine and inside current governance
- stop and report when one of the following is true:
  - a significant experimental breakthrough appears
  - a clearly acceptable engineering/code optimization progress is achieved
  - an escalation boundary in this file is triggered
  - user asks to stop or re-prioritize
