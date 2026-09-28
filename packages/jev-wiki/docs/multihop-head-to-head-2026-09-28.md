# jev-wiki against other memory designs on multi-hop questions (2026-09-28)

The [LongMemEval head-to-head](memory-head-to-head-2026-09-28.md) mostly asks for one
fact from one chat, so George pointed out that it tests search, not what makes an LLM
Wiki worth building: facts from many sources connected under shared pages. This runs the
same comparison on questions that need that.

## Verdict

**jev-wiki loses on connected questions at its defaults, and a simple fix gets it back
level with plain search. The LLM Wiki does worse than both.**

- Plain BM25 over article chunks answers 0.83 of the 80 questions. jev-wiki at the CLI
  defaults answers 0.76. BM25 alone got 10 right that jev-wiki missed, jev-wiki alone 4.
- The cause is what jev-wiki hands the reader: single sentences. A question like "who
  do both TechCrunch and The Verge say was behind X" needs the paragraph around each
  sentence, and sentence claims drop it. When each recalled claim is sent as the
  passage around it in its article, jev-wiki answers 0.80, and the gap to BM25 (5 only
  BM25, 3 only jev-wiki) is noise. Sending more sentences instead (0.79) helps less.
- Karpathy's LLM Wiki, where the model files every article's facts onto entity pages,
  answers 0.46 when the model picks pages from the index, as the pattern describes,
  and 0.71 when the pages are searched with BM25 instead. It is worst on exactly the
  connected questions: comparisons 0.15 browsed and 0.65 searched, against 0.85 for BM25.
  The structure it builds doesn't make up for what it loses in rewriting.
- Reading only the articles that hold the answer scores 0.86, so every design here but
  the browsed wiki is within a few questions of the ceiling this reader allows.

So the bet holds as "simple retrieval memory done well", not as a relations system. On
these questions no simple design gets anything from building relations up front: the
reader connects the facts itself when it is given the right passages. The one change
this points to is returning passages rather than lone sentences for document sources.
That is a product change to recall, so it isn't made here.

## Setup

- **Data**: MultiHop-RAG (Tang and Yang, COLM 2024): news articles from late 2023, with
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
| BM25 | the 20 best chunks of about 1,000 characters | none |
| **jev-wiki** | recalled claims at the CLI defaults (up to 40 for aggregate questions, neighbours, date windows) | local, 10 minutes for 150 articles |
| jev-wiki, more claims | up to 40 claims for every question | same |
| jev-wiki, passages | the same recall, each claim sent as the paragraphs around it (about 1,000 characters), overlaps merged, up to 20,000 characters | same |
| LLM Wiki | up to 10 pages the model picks from the index | 150 LLM calls, one after another |
| LLM Wiki, searched | whole pages in BM25 order, up to 20,000 characters | same |
| oracle | the question's evidence articles | not a real system |

The LLM Wiki reads each article next to the current index and files its facts, each
naming the source and date, onto new or existing pages. After 150 articles it has 2,624
pages holding 5,967 facts, 1,775 of the pages with a single fact. The index alone is
about 300,000 characters and every page together 1.07M, so the all-pages variant from
the LongMemEval run doesn't fit a prompt here.

## Results

Majority of 3 answers, 20 questions per type column. With no memory the reader answers
only the null questions (0.25 overall).

| System | All | Inference | Comparison | Temporal | Null | Notes sent (chars) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | **0.83** | 0.95 | 0.85 | 0.55 | 0.95 | 20,300 |
| jev-wiki | 0.76 | 0.80 | 0.74 | 0.55 | 0.95 | 10,700 |
| jev-wiki, more claims | 0.79 | 0.85 | 0.80 | 0.55 | 0.95 | 15,000 |
| **jev-wiki, passages** | **0.80** | 0.85 | 0.85 | 0.55 | 0.95 | 20,000 |
| LLM Wiki | 0.46 | 0.55 | 0.15 | 0.20 | 0.95 | 3,400 |
| LLM Wiki, searched | 0.71 | 0.85 | 0.65 | 0.40 | 0.95 | 20,000 |
| oracle | 0.86 | 1.00 | 0.80 | 0.65 | 1.00 | 20,300 |

One jev-wiki question hit a rate limit on every retry, so its row is out of 79.

Paired (right only in the other system / right only in the jev-wiki variant):

| Against | jev-wiki | jev-wiki, passages |
|---|---|---|
| BM25 | 10 / 4 | 5 / 3 |
| LLM Wiki, searched | 6 / 10 | 4 / 11 |
| LLM Wiki | 1 / 24 | 2 / 29 |
| oracle | 11 / 3 | 9 / 4 |

## Why each system lost what it lost

- **jev-wiki**: 7 of its 18 wrong answers on answerable questions are "Insufficient
  information". The claims found the right articles but a sentence like "The company
  said it would appeal" doesn't say which company. Passages carry that, and the
  comparison column goes from 0.74 to 0.85.
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

## What this means for sharing

Put jev-wiki forward as a cheap, local retrieval memory that matches plain search and
beats an LLM-built wiki, with the numbers from both runs. Don't put it forward as a
system that understands relations: on this test nothing simple does, and the fix that
worked was giving the reader more of the original text, not more structure.

Data: `/mnt/project-files/benchmarks/multihop-head-to-head-2026-09-28/` (`jev-bm25-oracle.json`
and `llm-wiki.json`). The LLM Wiki is cached in
`/mnt/project-files/benchmarks/llm-wiki-cache/multihop/`, so reruns only pay for
reading and judging.
