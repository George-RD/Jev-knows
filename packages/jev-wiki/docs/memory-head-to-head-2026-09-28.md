# jev-wiki against other memory designs on LongMemEval (2026-09-28)

George asked whether jev-wiki is genuinely useful memory next to the systems people rate
highest, Karpathy's LLM Wiki and cognee among them, or whether the bet is flawed. This
puts them through one test with everything else held fixed.

## Verdict

**Worth continuing and showing, with one quick fix first.**

- jev-wiki is the best of the practical designs here: 0.82 on answerable questions,
  within 3 points of reading only the right chats (0.85). It ties an LLM Wiki whose
  every page goes to the reader (0.81), while sending a ninth of the text and making
  no LLM calls to build or search its memory.
- The LLM Wiki as Karpathy describes it, where the model reads the index and picks
  pages, scores 0.71. Picking pages loses counting and date questions.
- Stuffing every chat into the prompt scores 0.67, worse than any retrieval.
- Real JEV intake keeps 29% of what it reads and answers as well as keeping
  everything (0.84 against 0.78 on the 30 cached questions; 3 answers better, none
  worse). JEV costs nothing in accuracy here, but this benchmark
  can't show what it gains either: a smaller store only pays off at a scale
  LongMemEval doesn't reach.
- The obvious fix is preference questions ("any tips for my guitar shopping?"). jev-wiki
  gets half of them; the LLM Wiki gets 85%, because its topic pages bring the user's
  tastes along even when the question doesn't name them.

What it doesn't show: jev-wiki beating plain BM25 over whole chats (0.77) by more
than the noise. The margin is 5 points; paired over both samples, jev-wiki alone got 13
answers right and BM25 alone got 8, which a sign test puts at p = 0.38.

## Setup

- **Data**: LongMemEval_S (Wu et al., ICLR 2025), each question with about 49 past
  chats. 60 questions, 10 of each type (seed 0, no abstention questions), plus the 30
  questions of the live sweep, whose JEV answers are already in the shared cache.
- **Same reader and judge for everyone**: `gpt-oss:120b` answers from the notes each
  system hands it; `glm-5.3` grades with LongMemEval's own judge prompts. Only the
  notes differ. Each set was run twice because the reader flips about 6% of verdicts
  between identical runs; the tables average the two.
- **User turns only**, as in every harness here. The `single-session-assistant` type
  asks what the assistant said, which no system stores, so the headline column leaves
  it out.
- Script: `examples/memory_head_to_head.py`.

The systems:

| System | What the reader gets | Cost to build and search |
|---|---|---|
| none | nothing | none |
| full context | every chat, dated (about 16k tokens) | none |
| BM25 chats | the 5 chats BM25 ranks top, dated | none |
| **jev-wiki** | recalled claims, dated, up to 40 for counting and date questions (CLI default) | local, about 10 s per 49 chats with the embedder |
| jev-wiki, live JEV | same recall over what JEV intake kept at the shipped gates | about 100 JEV requests per 49 chats |
| LLM Wiki | the pages the model picks from the index (up to 10) | 49 LLM calls to build (about 80 s), 1 to pick |
| LLM Wiki, all pages | every page | 49 LLM calls to build |
| oracle | only the chats that hold the answer | not a real system |

The LLM Wiki follows Karpathy's pattern with the same model: it reads each chat in date
order next to the current index (page titles and one-line summaries) and files the
chat's facts, dated, onto new or existing pages. The 60 wikis average 65 pages and 241
facts. It has no lint pass; the dated facts let the reader resolve updates.

## Results: 60 questions, two samples

Accuracy. "Answerable" leaves out `single-session-assistant`; each type column has 10
questions per sample.

| System | Answerable | Multi-session | Temporal | Knowledge update | Single-session user | Preference | Notes sent (chars) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| full context | 0.67 | 0.55 | 0.80 | 0.75 | 0.90 | 0.35 | 62,600 |
| BM25 chats | 0.77 | 0.50 | 1.00 | 0.90 | 0.95 | 0.50 | 8,900 |
| **jev-wiki** | **0.82** | 0.60 | 1.00 | 1.00 | 1.00 | 0.50 | **4,200** |
| LLM Wiki | 0.71 | 0.45 | 0.50 | 1.00 | 0.75 | **0.85** | 4,000 |
| LLM Wiki, all pages | 0.81 | 0.55 | 0.90 | 1.00 | 0.85 | 0.75 | 36,500 |
| oracle | 0.85 | 0.75 | 1.00 | 0.95 | 0.95 | 0.60 | 2,300 |

## Results: the 30 cached questions, with real JEV intake

Same setup on the live sweep's 30 questions (25 answerable, 5 per type), two samples.
JEV intake stored 531 claims per question on average and kept 155 active (29%) at the
shipped 0.82 gate. Every JEV answer came from the shared cache.

| System | Answerable | Multi-session | Temporal | Knowledge update | Single-session user | Preference |
|---|---:|---:|---:|---:|---:|---:|
| full context | 0.74 | 0.8 | 0.9 | 0.8 | 0.8 | 0.4 |
| BM25 chats | 0.74 | 0.7 | 0.5 | 1.0 | 1.0 | 0.5 |
| jev-wiki, keep everything | 0.78 | 0.6 | 0.8 | 1.0 | 0.8 | 0.7 |
| **jev-wiki, live JEV** | **0.84** | 0.8 | 0.8 | 1.0 | 0.8 | 0.8 |
| LLM Wiki | 0.68 | 0.3 | 0.6 | 0.9 | 0.8 | 0.8 |
| LLM Wiki, all pages | 0.78 | 0.4 | 0.8 | 1.0 | 0.8 | 0.9 |
| oracle | 0.88 | 0.8 | 0.8 | 1.0 | 1.0 | 0.8 |

With 5 questions per type, one question is 10 points in a type column. Read the
answerable column and the paired count: live JEV got 3 answers right that keep-everything
missed and lost none. Keeping 29% of the claims didn't cost answers; with less
competing text it may even help, but 3 questions is too few to claim that.

Paid TypeSafe requests: 100. 99 of them were a smoke test that ran one question outside
the cached set by mistake; the two runs of the 30 made none.

## What the other systems' published numbers mean

Reddit wouldn't load from here, so the post George saw couldn't be checked. The
published LongMemEval scores (Hindsight 91.4%, Zep 71.2%, and others) use different
readers and judges, include the assistant's turns, and often use the larger
LongMemEval_M. Hindsight's own paper puts full context at 39% on a 20B model; here the
same idea scores 0.67 because only the user's turns are sent and the reader is bigger.
They can't be lined up against the numbers above. What carries over is the pattern:
the systems at the top are retrieval systems with dated facts, not bigger context.
cognee is in this repository but needs an LLM pipeline per chat, so it wasn't run;
the LLM Wiki stands in for the "LLM builds a structured memory" family.

## Where jev-wiki loses

- **Preference (0.50).** Questions like "can you recommend a show tonight?" don't name
  what matters ("the user likes storytelling stand-up on Netflix"). jev-wiki recalls
  sentences that share the question's words, which here are generic. The LLM Wiki
  files that taste under a topic page, and the page picker finds the topic.
- **Multi-session counting (0.60 against the oracle's 0.75).** Known from
  [aggregate recall](aggregate-recall-default-2026-09-28.md): some mentions stay out
  of the notes.

## Next

Try one fix for preference questions, then rerun this comparison: when a question asks
for advice or recommendations, also recall the claims about the user's tastes in that
topic (for example, the best-matching chats' claims of kind preference). If that
closes most of the 35-point gap to the LLM Wiki, jev-wiki leads every simple design
here on every type, and that's the result to share.

Data: `/mnt/project-files/benchmarks/memory-head-to-head-2026-09-28/`. The LLM Wikis
are cached in `/mnt/project-files/benchmarks/llm-wiki-cache/`, so reruns only pay for
reading and judging.
