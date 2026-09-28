# Aggregate recall on by default, and dated recall (2026-09-28)

[Aggregate recall](aggregate-recall-2026-09-28.md) was measured before four review
fixes changed how its extra claims are packed. This reruns the QA benchmark on main,
turns aggregate recall on for CLI `recall`, and looks at the temporal questions that
still fail with every evidence chat recalled.

## Rerun on main

Same setup as before: all 500 LongMemEval_S questions, offline keep-everything intake,
embedder on, reader `gpt-oss:120b`, judge `glm-5.3`. The 20-claim notes on main match
the earlier run for 482 of 500 questions (the other 18 differ by a few characters of
embedding ties), so those answers were reused. Every question whose aggregate notes
differ from its 20-claim notes was answered and judged again.

| Type | n | 20 claims | up to 40 | up to 60 | earlier run: 40 / 60 |
|---|---:|---:|---:|---:|---:|
| multi-session | 121 | 0.719 | 0.785 | **0.802** | 0.777 / 0.818 |
| temporal-reasoning | 127 | 0.835 | **0.866** | 0.858 | 0.866 / 0.858 |
| knowledge-update | 72 | **0.917** | 0.903 | 0.889 | 0.931 / 0.903 |
| single-session-user | 64 | 0.953 | 0.953 | 0.953 | 0.953 / 0.953 |
| single-session-assistant | 56 | 0.321 | 0.321 | 0.339 | 0.304 / 0.321 |
| single-session-preference | 30 | 0.600 | 0.600 | 0.600 | 0.600 / 0.600 |
| abstention | 30 | **0.700** | **0.700** | 0.567 | 0.633 / 0.633 |
| **all** | 500 | 0.754 | **0.776** | 0.770 | 0.772 / 0.778 |

- The gain holds. Up to 40 turns 22 wrong answers right and 11 right answers wrong
  (sign test p = 0.08); up to 60 turns 21 right and 13 wrong (p = 0.23).
- The reader is not deterministic at temperature 0 on Ollama Cloud. Of 290 questions
  whose 40-claim notes were identical in both runs, 18 (6%) got a different verdict.
  Differences of one or two points between runs are noise, so read the two runs
  together.
- Averaged over both runs, 40 and 60 tie overall (0.774 each). 60 is 2 to 3 points
  better on multi-session counting. 40 is better on abstention (0.67 against 0.60)
  and knowledge updates, and its context is a third smaller (median 5.5k characters
  against 8.1k).

## The default

CLI `recall` now uses `--aggregate-limit 40` unless told otherwise, and
`--aggregate-limit 0` turns it off. 40 was picked over 60 because it ties overall at
two thirds of the context and hurts abstention less. `--max-chars` (default 6,000)
still bounds the context, so with CLI defaults a counting question gets as many
whole claims as fit in 6,000 characters, up to 40, instead of 5. The prompt hook is
unchanged, and `Engine.recall` still defaults to `aggregate_limit=None`.

## Temporal questions that fail with every evidence chat recalled

With up to 60 claims, 12 temporal questions were wrong although every evidence chat
had a claim in the notes. Checked turn by turn, they fall into four groups:

- **Evidence sentence not recalled (5).** The chat is there but not the sentence
  with the answer, such as "I got my own set of sculpting tools today" for "an
  investment for a competition four weeks ago", or the sentence dating the training
  pads. That's vocabulary, not arithmetic.
- **Relative dates resolved wrong (4).** "I mentioned visiting a museum two months ago"
  had the right note, but the reader picked a museum visit from another month that
  ranked higher; it has to date every note against the question date to find the one
  in range. Two more need a relative phrase inside a note ("started about a month
  ago", "booked three months in advance") anchored to that note's date.
- **Ordering over many dated notes (1).** The reader dropped one airline from a
  four-flight order.
- **Label problems (2).** 81 days between flu recovery and the tenth jog is 11.6
  weeks; the label says 15. "Exactly three weeks ago" is labelled two weeks.

So the failures past 40 claims aren't mostly date arithmetic. The group within
jev-wiki's reach is the second, and it pointed at a bigger gap: **`recall` returned no
dates at all.** The benchmark added chat dates to the notes itself, but an agent
calling `jev-wiki recall` had no way to answer "when" or "N weeks ago".

## Dated recall

- Every item carries `date`, and every context block shows it:
  `[1] raw/…#chars=0-80 (fact, 2023-05-16).` The date is `metadata["date"]` when the
  caller supplies one at ingest (an ISO 8601 date: when it happened), otherwise the
  capture time.
- `recall(..., as_of=...)` (default today, UTC) resolves a relative date in the query:
  "10 days ago", "two weeks ago", "a month ago", "yesterday" (and the day before),
  "last Saturday", "last
  weekend", "last week/month/year" (the calendar period), "in the past month" (up to
  today). Counted phrases allow rounding: a day either side for days, three for weeks,
  ten for months; named days allow one either side, since today and capture times are
  UTC. `time_window()` does this with one regex, no model call. The result
  reports the window as `time_window`.
- Claims from sources dated in that window are packed first, keeping their ranked
  order. No new candidates are added, and the JEV request is unchanged, but when the
  limit or `max_chars` is reached, in-window claims take the places of better-ranked
  claims from other dates. That's the intent: the question named the date.

On the 33 LongMemEval questions with such a phrase (up to 40 claims, three reader
samples each), accuracy went from 0.747 to 0.778. The notes changed for 23 of them.
Two questions went from mostly wrong to always right (the museum and art-event
questions above), and one went from always right to right two times in three. Seven
fail either way; five of them never recall the evidence sentence ("kitchen appliance"
for a smoker, "jewelry" for a chandelier), which reordering can't fix.

The QA harness now ingests each chat with its date and recalls as of the question's
date, so later runs include this.

## Next

- The largest remaining temporal and counting loss is the evidence sentence missing
  while its chat is recalled. For a dated question, the in-window claims most similar
  to the query could join the candidates even below the embedding threshold.

## Data

In the project's shared folder:
`/mnt/project-files/benchmarks/longmemeval-qa-aggregate-main-2026-09-28.jsonl` (one
row per question with the notes, answer and verdict for 20, 40 and 60 claims) and
`/mnt/project-files/benchmarks/longmemeval-qa-time-window-2026-09-28.jsonl` (the 33
dated questions, with and without the window).
