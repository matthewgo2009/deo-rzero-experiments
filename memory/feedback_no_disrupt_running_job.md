---
name: feedback-no-disrupt-running-job
description: "When a long experiment is running, edit files for the next run only; never restart the running Python process"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: aa7cad8c-70a1-4475-9eab-8b1ad118f39e
---

If a long-running experiment (DEO training run, verl GRPO loop, anything multi-hour) is
in flight, edits to its source files are SAFE only because the running Python process
already imported the module into memory — disk edits don't reach it. Don't even consider
SIGHUP / restart / `docker exec` to reload. The correct flow is:

1. Make file edits freely; current run continues unaffected.
2. Tell the user the new code only takes effect on the **next** run.
3. Plan when the next run starts (let current finish, or stop+archive+restart).

**Why:** Across the 2026-05-12 to 05-13 DEO session we accumulated iter 1/2/3 results
worth ~25 hours. The user values that data accumulation; restarting mid-run to apply
new code would have lost it. Even when iter 4 crashed on a verl Ray bug, we waited to
naturally exit before integrating filter changes.

**How to apply:**
- Before editing, check if anything is running (`docker ps`, `tmux ls`, log mtimes).
- If a run is active, say so explicitly: "改动落盘了,当前 run 不受影响,下次启动生效."
- For data backups before restart, suggest `sudo mv /eph/.../DEO /eph/.../DEO_archived_<TS>`
  (see DEPLOY.md "Restart fresh" section).

Related: [[feedback-data-driven-recommendations]].
