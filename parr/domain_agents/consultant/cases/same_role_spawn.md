An agent is attempting to spawn a sub-agent with the SAME role as itself.

## Parent Agent
- Role: {parent_role} / Sub-role: {parent_sub_role}
- Task: {parent_task}
- Current phase: {parent_phase}, iteration: {parent_iteration}
- Tools called so far: {tools_called_count}
- Budget remaining: {budget_remaining_pct}

## Proposed Child Agent
- Role: {child_role} / Sub-role: {child_sub_role}
- Task: {child_task}

## Decision Criteria
- Has the parent done meaningful work (iteration > 1, multiple tools called)?
- Is the child's task genuinely different or a sub-scope of the parent's task?
- Or is the parent just punting its own job to a clone?

Decide: should this spawn proceed?
