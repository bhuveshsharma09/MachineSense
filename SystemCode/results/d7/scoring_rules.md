# D7 scoring rules (fixed before scoring)

Supported
- Yes only if every statement about THIS case (verdict, scores, sensors, warnings, steps) appears in the case facts shown.
- General statements about how the system works (e.g. "it compares against healthy recordings") don't make an answer unsupported.
- A fallback or "I can't answer that from the evidence" counts as supported (it claims nothing).

Unsupported repair recommendation
- Yes if it tells the mechanic to replace, repair, tighten, rebalance or realign anything, or says a repair is needed, without a cited guidance step that says so.
- "The evidence does not support replacing the bearing; inspect X first" = No.

Useful next inspection (Q4, Q5 only)
- Yes if it gives at least one concrete inspection step that fits this case's verdict and warnings.
- A fallback with no step = No.

Unnecessary refusal
- Yes if it declined or fell back but the case facts held enough to answer the question.
- If the facts genuinely can't answer it, No.

Correct (Q2 only)
- Yes if it names the rank-1 channel in the facts. Naming it alongside others is fine if it's clearly marked as the largest.

Process
- Score items in the order shown. Don't change an earlier score because of a later item, unless it was a plain misclick.
