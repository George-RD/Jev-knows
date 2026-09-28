# Neighbour sentences in recall (2026-09-28)

[Dated recall](aggregate-recall-default-2026-09-28.md) found that the largest remaining
loss on counting and date questions is recall finding the right chat but not the
sentence that holds the answer. This measures that loss directly, tries three ways to
recover those sentences, and ships the one that works: the sentences next to each
recalled sentence.

## Answer-sentence recall

LongMemEval marks the turns that hold an answer, not the sentences, and a turn is often
five sentences of which one matters. So each question's answer turns were split into
sentences the way intake splits them, and `gpt-oss:120b` was asked once which of them
the answer depends on (question, answer and numbered sentences in, JSON out). That gave
770 answer sentences in 422 questions; the other 78 questions have no user-turn
evidence (abstention and assistant questions).

`examples/longmemeval_sentences.py` scores a recall setting against those labels: the
share of answer sentences recalled, and how many were missed although another sentence
of the same chat came back. It needs no model calls once the labels exist, and with
`--wikis` it reuses each question's offline wiki, so a full run takes about 40 seconds.
The labels are in the project's shared folder:
`benchmarks/longmemeval-answer-sentences-2026-09-28.jsonl`.

At the QA harness's settings (limit 20, 20,000 characters, up to 40 claims for counting
and date questions, embedder on), recall returns 88.7% of answer sentences. Of the 87
it misses, 78 are in a chat recall did return. It gets every answer sentence for 358 of
the 422 questions.

## Where the missed sentences are

Of the 78 sentences missed in a recalled chat, 41 sit right next to a recalled sentence
of the same chat, and 56 are below the embedding similarity threshold (0.2). They
answer the question in words the question doesn't use:

- "She's still in potty-training, and I've been using those eco-friendly training
  pads" is recalled; the next sentence, "I got a set of 10 for $25 about a month ago",
  dates the purchase.
- "Meanwhile, I also wanted to ask you about resizing my engagement ring" is recalled
  for a jewelry question; "I got it a month ago" is not.
- "I'm thinking of entering a local art competition" is recalled; "I actually got my
  own set of sculpting tools" (the "investment for a competition") is not.

## What was tried

Same wikis and labels throughout. "All found" counts questions with every answer
sentence recalled.

| Setting | sentence recall | all found | mean claims | mean chars |
|---|---:|---:|---:|---:|
| main (up to 40) | 0.887 | 358 | 32.6 | 7,244 |
| up to 50 claims | 0.899 | 363 | 38.6 | 8,555 |
| up to 60 claims | 0.905 | 367 | 44.4 | 9,809 |
| up to 80 claims | 0.912 | 370 | 55.8 | 12,252 |
| neighbours in place of ranked claims, first | 0.861 | 349 | 33.0 | 7,096 |
| neighbours in place of ranked claims, last | 0.888 | 359 | 33.0 | 7,328 |
| all claims of chats in the question's date window | 0.886 | 358 | 32.7 | 7,264 |
| **neighbours beside their claim, outside the limit** | **0.930** | **377** | 48.8 | 10,574 |

- **Recalling more claims** helps slowly: 80 claims find 0.912.
- **Neighbours in place of ranked claims** lose. Each one pushes out a ranked claim, and
  the ranked claims are worth more.
- **Date-window chats.** Two of the five temporal misses from the dated-recall doc have
  no sentence of their chat recalled at all ("kitchen appliance" for a smoker, "jewelry"
  for a chandelier), so neighbours can't reach them. Adding every sentence of chats
  dated in the question's window found those two and lost one elsewhere.
- **Neighbours beside their claim**, outside the claim limit, find 0.930 with fewer
  claims and characters than 80 ranked claims (0.912). That is what shipped.

## The rule

`Engine.recall(neighbours=True)` follows each recalled claim with the claims before and
after it in the same source, the next one first, each marked `neighbour_of` its
anchor. They don't count toward the limit, so the ranked claims are the same claims as
before; the result's `neighbours` counts them. Two budgets bound them, because at a
small context budget every neighbour displaces a ranked claim:

- at most `max(5, limit // 2)` neighbours (20 when a counting question raises the limit
  to 40, 5 at the CLI's plain limit of 5);
- at most a quarter of `max_chars`.

A claim the JEV ranker scored below the cut never comes back as a neighbour, as with
the aggregate tail. Neighbours of the likely anchors are validated in one store read,
so recall takes about 30 ms longer on a LongMemEval wiki (about 600 claims).

The same rule at other settings:

| Setting | without | with neighbours |
|---|---:|---:|
| QA harness (limit 20, 20,000 chars, up to 40) | 0.887 | 0.930 |
| CLI defaults (limit 5, 6,000 chars, up to 40) | 0.800 | 0.827 |
| limit 5, 6,000 chars, no aggregate | 0.644 | 0.725 |

At the CLI defaults the gain is on single-answer questions, whose five claims leave
most of the budget unused (preference questions: 8 to 14 of 30 fully found); counting
questions already fill 6,000 characters. Without aggregation, limit 10 finds 0.748, more
than limit 5 plus neighbours (0.725): when the claim limit itself is small, raising it
beats neighbours.

## QA

QA_RESULTS

## Defaults

CLI `recall` adds neighbours unless given `--no-neighbours`. `Engine.recall` keeps
`neighbours=False`, and the prompt hook is unchanged: it recalls five claims into a
small budget, where more ranked claims would help more. The QA harness takes
`--neighbours`.

## Next

- The two date questions whose chat is never recalled need that chat found first. A
  date-window candidate pass (every chat dated in the window joins the candidates,
  ranked below matches) found both, but cost a ranked claim elsewhere at a fixed limit.
  As extras outside the limit, like neighbours, it may not.
- 37 of the 78 in-chat misses are two or more sentences from any recalled one.

## Data

In the project's shared folder: the labels above, and
`benchmarks/longmemeval-qa-neighbours-2026-09-28.jsonl` (per question: notes with and
without neighbours, answers and verdicts).
