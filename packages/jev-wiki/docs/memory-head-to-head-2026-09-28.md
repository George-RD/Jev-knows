# jev-wiki against other memory designs on LongMemEval (2026-09-28)

George asked whether jev-wiki is genuinely useful memory next to the systems people rate
highest, Karpathy's LLM Wiki and cognee among them, or whether the bet is flawed. This
puts them through one test with everything else held fixed.

## Verdict

**Worth sharing.** At its CLI defaults, jev-wiki gives the best answers of every
practical design tested here, while sending the reader a sixth of the text of the
best LLM Wiki and making no LLM calls to build its memory.

- On 60 questions jev-wiki scores 0.82 on answerable questions, against BM25 over whole
  chats 0.78, an LLM Wiki with every page in the prompt 0.74, every chat in the prompt
  0.70, and Karpathy's pattern with the model picking pages 0.66. Reading only the
  chats that hold the answer scores 0.88.
- With real JEV intake, on the 30 questions whose JEV answers are cached, it scores
  0.96 against 0.88 for keeping everything. JEV kept 29% of the claims it read. It got
  2 answers right that keep-everything missed and lost none.
- The preference gap from the first run is gone. jev-wiki now answers 11 of the 15
  preference questions, the LLM Wiki 12 of 15 when it picks pages and 10 of 15 when it
  sends them all. Neighbour claims (PR #28) did it. The fix tried here, searching with
  the topic words only, found no more evidence and was dropped (see below).

What it doesn't show: jev-wiki beating plain BM25 by more than noise. On answerable
questions over both sets jev-wiki alone got 10 right and BM25 alone got 7. The claim
that stands up is "as good as or better than every simple design, at a fraction of the
context". Only the gap to the page-picking LLM Wiki (22 against 8) and to full context
(12 against 4) is clear.

## Setup

- **Data**: LongMemEval_S (Wu et al., ICLR 2025), each question with about 49 past
  chats. 60 questions, 10 of each type (seed 0, no abstention questions), plus the 30
  questions of the live sweep, whose JEV answers are already in the shared cache.
- **Same reader and judge for everyone**: `gpt-oss:120b` answers from the notes each
  system hands it; `glm-5.3` grades with LongMemEval's own judge prompts. Only the
  notes differ. Each question is answered 3 times from the same notes and graded by
  majority, because the reader flips about 6% of verdicts between identical runs.
- **User turns only**, as in every harness here. The `single-session-assistant` type
  asks what the assistant said, which no system stores, so the headline column leaves
  it out.
- Script: `examples/memory_head_to_head.py`.

The systems:

| System | What the reader gets | Cost to build and search |
|---|---|---|
| full context | every chat, dated (about 16k tokens) | none |
| BM25 chats | the 5 chats BM25 ranks top, dated | none |
| **jev-wiki** | recalled claims, dated, at the CLI defaults: up to 40 for counting and date questions, each with its neighbouring claims, plus claims from a named date window | local, about 10 s per 49 chats with the embedder |
| jev-wiki, live JEV | same recall over what JEV intake kept at the shipped gates | about 100 JEV requests per 49 chats |
| LLM Wiki | the pages the model picks from the index (up to 10) | 49 LLM calls to build (about 80 s), 1 to pick |
| LLM Wiki, all pages | every page | 49 LLM calls to build |
| oracle | only the chats that hold the answer | not a real system |

The LLM Wiki follows Karpathy's pattern with the same model: it reads each chat in date
order next to the current index (page titles and one-line summaries) and files the
chat's facts, dated, onto new or existing pages. The 60 wikis average 65 pages and 241
facts. It has no lint pass; the dated facts let the reader resolve updates.

## Results: 60 questions

Majority of 3 answers. "Answerable" leaves out `single-session-assistant`; each type
column has 10 questions. With no memory the reader answers none.

| System | Answerable | Multi-session | Temporal | Knowledge update | Single-session user | Preference | Notes sent (chars) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full context | 0.70 | 0.6 | 0.8 | 0.8 | 1.0 | 0.3 | 62,600 |
| BM25 chats | 0.78 | 0.5 | 1.0 | 0.9 | 0.9 | 0.6 | 8,900 |
| **jev-wiki** | **0.82** | 0.5 | 1.0 | 0.9 | 1.0 | 0.7 | 6,100 |
| LLM Wiki | 0.66 | 0.5 | 0.5 | 0.8 | 0.8 | 0.7 | 3,900 |
| LLM Wiki, all pages | 0.74 | 0.4 | 0.9 | 1.0 | 0.8 | 0.6 | 36,500 |
| oracle | 0.88 | 1.0 | 1.0 | 0.9 | 1.0 | 0.5 | 2,300 |

Paired against jev-wiki on answerable questions (right only there / right only in
jev-wiki): BM25 4 / 6, LLM Wiki all pages 5 / 9, full context 2 / 8, LLM Wiki 7 / 15.

## Results: the 30 cached questions, with real JEV intake

Same setup on the live sweep's 30 questions (25 answerable, 5 per type). JEV intake
stored 531 claims per question on average and kept 155 active (29%) at the shipped
0.82 gate.

| System | Answerable | Multi-session | Temporal | Knowledge update | Single-session user | Preference |
|---|---:|---:|---:|---:|---:|---:|
| full context | 0.80 | 1.0 | 0.8 | 1.0 | 1.0 | 0.2 |
| BM25 chats | 0.84 | 0.8 | 0.8 | 1.0 | 1.0 | 0.6 |
| jev-wiki, keep everything | 0.88 | 0.8 | 1.0 | 1.0 | 0.8 | 0.8 |
| **jev-wiki, live JEV** | **0.96** | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 |
| LLM Wiki | 0.64 | 0.2 | 0.6 | 0.8 | 0.6 | 1.0 |
| LLM Wiki, all pages | 0.84 | 0.6 | 0.8 | 1.0 | 1.0 | 0.8 |
| oracle | 0.88 | 0.8 | 0.8 | 1.0 | 1.0 | 0.8 |

With 5 questions per type, one question is 20 points in a type column; read the
answerable column. Live JEV got 2 answers right that keep-everything missed and lost
none. So keeping 29% of the claims costs no answers here, and less competing text may
help, but 2 questions are too few to claim that.

Paid TypeSafe requests over the whole study: 102. 99 were a smoke test that ran one
question outside the cached set by mistake; a second smoke test and the three runs of
the 30 made the other 3.

## First run: before neighbours, one answer per question

The first run (PR #27) used recall without neighbours or window claims and one answer
per question, averaged over two runs. Answerable accuracy on the 60: jev-wiki 0.82,
LLM Wiki all pages 0.81, BM25 0.77, LLM Wiki 0.71, full context 0.67, oracle 0.85. The
difference that mattered was preference: jev-wiki 0.50 against the LLM Wiki's 0.85.
Majority voting shrank the LLM Wiki's preference lead, and neighbours raised
jev-wiki's. Data: `h2h60.json`, `h2h60b.json`, `h2h30.json`, `h2h30b.json`.

## The preference fix that didn't help

Advice questions ("any tips on what to bake?") recalled other times the user asked for
tips, because "tips", "recommend" and "suggestions" matched every such request. Searching
with the topic words only ("inviting colleagues small gathering bake") looked like the
fix. A free scorer over all 30 preference questions in LongMemEval_S (does a recalled
claim overlap the labelled answer turn?) found the evidence in 24 of 30 either way:
2 gained and 2 lost, and "recommend a show tonight" lost everything but one claim. The
evidence was already in the notes. The reader wasn't using it, and neighbours, which
give each recalled sentence the sentences around it, fixed that. The change was not
kept.

## What the other systems' published numbers mean

Reddit wouldn't load from here, so the post George saw couldn't be checked. The
published LongMemEval scores (Hindsight 91.4%, Zep 71.2%, and others) use different
readers and judges, include the assistant's turns, and often use the larger
LongMemEval_M. Hindsight's own paper puts full context at 39% on a 20B model; here the
same idea scores 0.70 because only the user's turns are sent and the reader is bigger.
They can't be lined up against the numbers above. What carries over is the pattern:
the systems at the top are retrieval systems with dated facts, not bigger context.
cognee is in this repository but needs an LLM pipeline per chat, so it wasn't run;
the LLM Wiki stands in for the "LLM builds a structured memory" family.

## Where jev-wiki still loses

- **Multi-session counting (0.5 against the oracle's 1.0 on the 60).** Some mentions
  stay out of the notes, as in [aggregate recall](aggregate-recall-default-2026-09-28.md).
  Every practical design here is at 0.4 to 0.6 on it.

Data: `/mnt/project-files/benchmarks/memory-head-to-head-2026-09-28/`
(`cli-defaults-3samples-60.json` and `cli-defaults-3samples-30.json` for the tables
above). The LLM Wikis are cached in `/mnt/project-files/benchmarks/llm-wiki-cache/`, so
reruns only pay for reading and judging.
