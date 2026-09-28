# Window claims in recall (2026-09-28)

[Neighbour sentences](evidence-neighbours-2026-09-28.md) recover an answer sentence
next to a recalled one. They can't help when no sentence of the answering chat is
recalled at all. Two dated questions failed that way: "What kitchen appliance did I buy
10 days ago?" against "I just got a smoker today" in a chat about BBQ sauce, and "I
received a piece of jewelry last Saturday from whom?" against a crystal chandelier from
an aunt. Neither sentence shares a word with its question, and both are below the
embedding threshold (0.13 and 0.09 against 0.2). What they share with the question is
the date.

## The rule

`Engine.recall(window_claims=True)`: when the query names a date ([dated
recall](aggregate-recall-default-2026-09-28.md) resolves "10 days ago", "last
Saturday" and so on), claims from sources dated inside that window that ranking left out
are added after all ranked claims, most similar to the query first (source order
without an embedder). Each is marked `in_window`, and the result counts them in
`window_claims`. Like neighbours they sit outside the claim limit: at most 20 of them,
in a quarter of `max_chars`. Because they come last, they only use room the ranked
claims and their neighbours left, so no ranked claim is displaced. A claim the JEV
ranker scored below the cut is never added. Without a named date nothing changes.

An earlier try put the same claims inside the limit, and it lost as much as it gained:
each one pushed out a ranked claim ([neighbour doc](evidence-neighbours-2026-09-28.md),
"What was tried"). A first version of this rule packed them straight after the window's
ranked claims, and review found that at a tight `max_chars` they could still crowd out
later ranked claims. Packing them last fixed that and found more.

## Answer-sentence recall

`examples/longmemeval_sentences.py` at the QA harness's settings (limit 20, 20,000
characters, up to 40 claims, embedder on, neighbours on):

| | sentence recall | all found | temporal all found | mean claims |
|---|---:|---:|---:|---:|
| neighbours | 0.930 | 377 | 117/130 | 48.8 |
| neighbours + window claims | **0.939** | **383** | **121/130** | 50.0 |

28 questions change, the ones whose window held claims that ranking left out. Six more
questions have every answer sentence found, the smoker and the chandelier among them,
and none loses one. Sentences missed with their chat unrecalled fall from 9 to 5. At the
CLI defaults (limit 5, 6,000 characters) it finds one more sentence (0.827 to 0.829):
the budget is mostly spent by then.

## QA

Those 28 questions through the QA harness's reader and judge, three samples per arm:

| | correct (mean of 3) |
|---|---:|
| neighbours | 0.690 |
| neighbours + window claims | **0.774** |

- Fixed in every sample: the smoker, eac54add. Mostly fixed: the chandelier and
  gpt4_e061b84f (0 to 2 of 3), and gpt4_1e4a8aec (2 to 3 of 3).
- Worse: 60159905 (3 to 1 of 3), 0bc8ad93 (a museum visit "two months ago", 2 to 0 of
  3), gpt4_59149c78 (3 to 2 of 3). Extra dated claims give the reader more to pick the
  wrong one from.
- gpt4_fa19884d shows the reader's noise: on identical notes without window claims it
  was right three times of three in a first run and wrong three times of three in this
  one.

Net that's about +2.3 of the 28 questions, or +0.5 points over all 500 LongMemEval
questions: small, but a class of question neighbours can't reach.

## Defaults

CLI `recall` adds window claims unless given `--no-window-claims`. `Engine.recall`
keeps `window_claims=False`, and the prompt hook is unchanged. The QA harness and the
sentence scorer take `--window-claims`.

## Next

- "I mentioned an investment for a competition four weeks ago" still fails. Its window
  (±3 days) holds several chats, and the sculpting-tools sentence ranks below 20 by
  similarity.

## Data

In the project's shared folder: `benchmarks/longmemeval-qa-window-claims-2026-09-28.jsonl`
(the 28 questions, both arms, three samples, notes, answers and verdicts). The earlier
`longmemeval-qa-time-window-2026-09-28.jsonl` is dated recall's run, not this one.
