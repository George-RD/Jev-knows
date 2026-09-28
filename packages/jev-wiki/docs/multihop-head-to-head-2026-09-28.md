# jev-wiki against other memory designs on multi-hop questions (2026-09-28)

The [LongMemEval head-to-head](memory-head-to-head-2026-09-28.md) mostly asks for one
fact from one chat, so George pointed out that it tests search, not what makes an LLM
Wiki worth building: facts from many sources connected under shared pages. This runs the
same comparison on questions that need that.

## Verdict

**On connected questions jev-wiki ties with plain search, and the LLM Wiki loses to
both.**

- BM25 over article chunks answers 0.81 of the 80 questions. jev-wiki answers 0.79,
  and 0.80 when each recalled claim is sent as the passage around it. The paired
  counts (8 right only in BM25, 6 only in jev-wiki) are noise: the same jev-wiki notes
  graded twice gave 0.76 and 0.79.
- Karpathy's LLM Wiki, where the model files every article's facts onto entity pages,
  answers 0.46 when the model picks pages from the index, as the pattern describes,
  and 0.71 when its pages are searched with BM25 instead. It does worst on exactly
  the connected questions: comparisons 0.15 browsed and 0.65 searched, against 0.85
  for jev-wiki. jev-wiki beats it 27 questions to 1 browsed and 11 to 5 searched.
- Reading only the articles that hold the answer scores 0.86, so jev-wiki and BM25 are
  within a few questions of the ceiling this reader allows.

So the bet holds as "simple retrieval memory done well", and not as a relations system.
On these questions, building relations up front gained nothing for any simple design:
the reader connects the facts itself when it gets the right text, and a wiki that
rewrites the articles into pages loses the details the questions turn on. jev-wiki's
own weak spot here is small. A lone sentence like "the company said it would appeal"
doesn't say which company. Sending passages instead of sentences fixes that for
comparisons (0.74 to 0.85 in the first run), but overall it is worth no more than a
point or two. Nothing about that points to changing recall.

## Setup

- **Data**: MultiHop-RAG (Tang and Yang, COLM 2024), news articles from late 2023, with
  questions that each need 2 to 4 of them. Four types: inference (what links two
  reports), comparison (do two sources agree), temporal (did coverage change between
  two dates), and null (the corpus can't answer). 80 questions, 20 of each type
  (seed 0).
- **One shared corpus**, as in a real knowledge base: the 80 questions' evidence
  articles plus seeded distractors, 150 articles and 1.7M characters, oldest first. No
  system can read all of it at once, so there is no full-context row.
- **Same reader and judge as before**: `gpt-oss:120b` answers only from the notes each
  system sends and must say "Insufficient information" otherwise; `glm-5.3` grades
  with LongMemEval's judge prompts (its abstention prompt for null questions). Each
  question is answered 3 times and graded by majority.
- Script: `examples/multihop_head_to_head.py`. No TypeSafe requests: jev-wiki keeps
  every claim offline, and the LLM Wiki is built with the reader model.

The systems:

| System | What the reader gets | Cost to build |
|---|---|---|
| BM25 | the 20 best chunks of about 1,000 characters, scored with their source and title | none |
| **jev-wiki** | offline recall as in the LongMemEval harness (up to 20 claims, 40 for aggregate questions, with neighbours and date windows), as of the newest article | local, 10 minutes for 150 articles |
| jev-wiki, more claims | up to 40 claims for every question | same |
| jev-wiki, passages | the same recall, each claim sent as the paragraphs around it (about 1,000 characters), overlaps merged, up to 20,000 characters | same |
| LLM Wiki | up to 10 pages the model picks from the index | 150 LLM calls, one after another |
| LLM Wiki, searched | whole pages in BM25 order, up to 20,000 characters | same |
| oracle | the question's evidence articles | not a real system |

The LLM Wiki reads each article next to the current index and files its facts, each
naming the source and date, onto new or existing pages. No article failed. After 150
articles it has 2,624 pages holding 5,967 facts, 1,775 of the pages with a single fact.
The index alone is about 300,000 characters and every page together 1.07M, so the
all-pages variant from the LongMemEval run doesn't fit a prompt here.

## Results

Majority of 3 answers, 20 questions per type column. With no memory the reader answers
only the null questions (0.25 overall).

| System | All | Inference | Comparison | Temporal | Null | Notes sent (chars) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | **0.81** | 0.95 | 0.75 | 0.60 | 0.95 | 20,100 |
| **jev-wiki** | **0.79** | 0.80 | 0.85 | 0.55 | 0.95 | 10,700 |
| jev-wiki, more claims | 0.80 | 0.90 | 0.75 | 0.60 | 0.95 | 15,000 |
| jev-wiki, passages | 0.80 | 0.85 | 0.85 | 0.55 | 0.95 | 18,900 |
| LLM Wiki | 0.46 | 0.55 | 0.15 | 0.20 | 0.95 | 3,400 |
| LLM Wiki, searched | 0.71 | 0.85 | 0.65 | 0.40 | 0.95 | 20,000 |
| oracle | 0.86 | 1.00 | 0.80 | 0.65 | 1.00 | 20,300 |

Paired against jev-wiki (right only in the other system / right only in jev-wiki):
BM25 8 / 6, jev-wiki with passages 4 / 3, LLM Wiki searched 5 / 11, LLM Wiki 1 / 27,
oracle 10 / 4.

**How much of this is noise.** The review run below regraded jev-wiki from exactly the
same notes, and 8 of 80 verdicts flipped even with 3 answers per question (0.76 then
0.79). So a gap of 2 or 3 points between two systems here means nothing; the LLM
Wiki's 8 to 33 point gaps do.

## Why each system lost what it lost

- **jev-wiki**: 16 wrong answers on answerable questions, 5 of them "Insufficient
  information", where the recalled sentences named the right articles but left out
  who or what they were about. The rest are wrong readings, mostly of temporal
  questions.
- **LLM Wiki, browsed**: 35 of its 42 wrong answers are "Insufficient information".
  The model picks about 2 pages out of 2,624 from a 300,000-character index, and the
  facts a question needs are spread over pages it didn't pick: "OpenAI", "OpenAI
  board", "Sam Altman – OpenAI Board" and "OpenAI Board Veto Power" are four of them.
  Karpathy's pattern has a lint pass to merge pages like these, which isn't built here,
  but an index this size is past what page picking can use either way.
- **LLM Wiki, searched**: finding pages by search fixes most of that, but a rewritten
  fact loses details the questions turn on (who reported what, and when), so temporal
  questions stay at 0.40.
- **Temporal questions** are hard for everyone, the oracle included (0.65): they ask
  whether two dated reports agree, which the reader often gets wrong with the right
  articles in front of it.

## First run and what the review changed

The first run gave BM25 0.83 against jev-wiki 0.76, and read as jev-wiki losing. A
review of the script then found two things that favoured one side or the other:
BM25 wasn't scoring the source and title line that jev-wiki scores with every claim,
and jev-wiki recalled as of today rather than as of the corpus. Both were fixed and
BM25 and every jev-wiki variant rerun. The date fix changed no jev-wiki notes; the
title fix moved BM25 from 0.83 to 0.81. jev-wiki went from 0.76 to 0.79 from the same
notes, which is the grading noise above. Passages scored 0.80 both times.

## What this means for sharing

Put jev-wiki forward as a cheap, local retrieval memory that matches plain search on
both single-fact and connected questions, and clearly beats an LLM-built wiki on
connected ones, with the numbers from both runs. Don't put it forward as a system that
understands relations: on this test nothing simple does, and what helps the reader is
the original text, not more structure.

Data: `/mnt/project-files/benchmarks/multihop-head-to-head-2026-09-28/multihop-80.json`
(the tables above; `first-run-*.json` for the first run). The LLM Wiki is cached in
`/mnt/project-files/benchmarks/llm-wiki-cache/multihop/`, so reruns only pay for
reading and judging.
