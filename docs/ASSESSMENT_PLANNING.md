# Team planning prompts — not submission prose

## Preliminary submission

- Verify the live demonstration date in Canvas and subtract four business days using the facilitator's definition and relevant holidays.
- Submit your own in-progress report; explicitly distinguish implemented, measured and planned work.
- Record 5–10 minutes of actual current team progress.
- Include prototype files, configuration, executed logs and available screenshots.
- Attach genuine week-by-week records; do not backdate or invent work.
- Inform the facilitator after the Canvas submission. A message has not been sent by this project.
- Complete the required Week 8–9 and Week 10–12 peer reviews.

The supplied brief states a 20% overall teamwork grade deduction without the preliminary submission. It contains conflicting live timing: 15 minutes plus 10-minute questions versus a template of 20 minutes plus 5–10-minute questions. Confirm the applicable timing.

## Suggested 7-minute progress recording sequence

This is a recording outline, not a voiceover script or an existing recording.

| Time | Evidence to show | Explain in your own words |
| --- | --- | --- |
| 0:00–0:45 | RQ and task list | What is being compared and what your team has completed |
| 0:45–1:45 | Raw schema, manifest, splits | Why graph-level labels and fixed partitions are appropriate |
| 1:45–3:15 | Model code and actual run log | GraphSAGE vs masked attention; who implemented or reviewed what |
| 3:15–4:45 | Saved metrics and figures | Actual results, confusion counts and one limitation |
| 4:45–6:00 | Local demo and checkpoint command | Evidence playback vs an actual forward pass; explain one graph |
| 6:00–7:00 | Genuine progress records | Current blockers, next experiments, responsibilities |

Record your own screen and team explanations. No team recording has been manufactured in the ZIP.

## 18-slide planning outline

Use these prompts to author your own slides; the project does not include finished assessment slides or a report.

1. Title and team; use the unit shown in your design PDF (COS60011), not a guessed unit.
2. Motivation: a cited real attack example and relevance to provenance.
3. Detection challenge and client deliverables.
4. Research gap and retained RQ1.
5. Related-work table with assumptions, supervision and output granularity.
6. Dataset choice and measured sample statistics.
7. Shared graph preparation and partitions.
8. FLASH-inspired implementation and omissions.
9. MAGIC-inspired implementation and omissions.
10. Baselines and why they are useful.
11. Experiment settings, dependencies, seeds and threshold rule.
12. Prototype demonstration.
13. Detection results and uncertainty across seeds.
14. False positives, missed attacks and plausible competing explanations.
15. Runtime/memory evidence and measurement limits.
16. Weekly progress and actual contributions.
17. Limitations, future work and what the team learned.
18. Evidence-based take-home message and references.

Prepare a 15-minute core presentation with optional detail until timing is confirmed. Share speaking time equally. Rehearse the demo offline, keep figures as backup, and test the actual presentation machine.

## Interview preparation questions

- Why does an attack graph not mean every node is malicious?
- Why can scenario/graph IDs not be input features?
- Which components retain each paper's central idea, and which are omitted?
- Why train MAGIC-inspired on benign data, and how do labelled validation data affect the unsupervised claim?
- Why AP/FPR rather than accuracy alone?
- What prevents test-label tuning in this code? What does it not prevent?
- Could a simple baseline win, and what would that imply?
- What does edge sampling remove, and can the label remain valid after sampling?
- What would be needed to migrate to DARPA TC, exact baselines or temporal detection?
- Can another member rerun your settings and explain your output files?

## Feedback for the supplied design PDF

- Confirm the chosen dataset and graph-level detection target in your own report.
- Correct repeated table numbering (team responsibilities and success criteria both labelled Table 8).
- Refresh the table of contents after final pagination.
- Avoid describing unexecuted plans as demonstrated results.
- Check that the preliminary report deadline precedes the actual live demonstration by four business days; the design plan groups both on 30 October.
- Retain only factual contribution claims verified by the team.
