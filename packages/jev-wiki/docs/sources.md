# Primary sources and evidence boundaries

Reviewed on **2026-09-24**. These links support the design; they are not a claim that
the package reproduces every capability mentioned. Documentation URLs can change.
The review date is a reading snapshot, not an immutable archive or content hash.
No third-party model guide or marketing-site replica was used as API authority.

## Karpathy LLM Wiki

- Author: Andrej Karpathy.
- Source: [LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
- Gist creation shown: 2026-04-04; one revision shown at review.
- Supporting facts: immutable raw evidence, generated Markdown wiki, schema
  instructions, ingest/query/lint operations, index and chronological log. Git and
  optional search tools are discussed. The gist intentionally leaves implementation
  details to each user and agent.
- Our additions: canonical JSON decisions, automatic capture, JEV selectors,
  revision-aware transitions, bounded recall, tombstones and deterministic rendering.

## TypeSafe primitives and API

- [Introduction](https://docs.typesafe.ai/introduction).
- [API reference](https://docs.typesafe.ai/api).
- [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out).
- Supporting facts: `POST https://api.typesafe.ai/v1/systemone` accepts `state`,
  `model` and a map of typed questions. Choice selects from supplied options; Score
  evaluates an ordered rubric; Noul reports a yes/no probability. Multiple questions
  share state but are evaluated independently. Question map keys identify answers;
  they are not model instructions.
- Design consequence: semantics belong in each question's instructions/criteria.
  Application code must validate returned types, IDs and bounds before applying
  decisions. Batching does not make dependent questions sequential.

## Extraction and evidence

- [Pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook).
- [Citation checking](https://docs.typesafe.ai/cookbooks/citation_check).
- Supporting facts: the extraction recipe first discovers candidate values in code,
  then asks JEV to select among them, and copies the selected value verbatim. The
  citation recipe judges a claim against supplied source context.
- Our choice: paragraph candidates preserve broader evidence than isolated values.
  Exact textual matching proves where an excerpt came from; neither an exact match
  nor a model judgment establishes that the source is factually correct.

## Confidence

- [Confidence documentation](https://docs.typesafe.ai/confidence).
- Supporting facts: Choice and Score confidence is derived from their probability
  distributions. Noul has no separate confidence field. Thresholds depend on task
  consequences and observed domain performance.
- Our choice: thresholds are policies requiring validation. They are not truth
  scores, and repeated judgments do not count as new corroborating sources.

## Known model limitations

- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
- Provider page's own review date: 2026-09-17.
- Supporting facts: documented weaknesses include literal interpretation, arithmetic,
  date comparison, indirection, irrelevant context and adversarial content. Jev is
  not trained for prose generation. The provider recommends precise questions,
  bounded state and code for exact computation.
- Our choice: these are architectural constraints, not issues solved by a larger
  prompt. Memory injection remains a data-trust boundary.

## Model identity and limits

- [Models](https://docs.typesafe.ai/models).
- Documented stable version at review: `jev-1.13.0`; moving aliases include
  `jev-latest` and `jev-preview`.
- Documented limits at review: 64k total request tokens and 32k for state plus the
  longest question. These are upper bounds, not recommended batch sizes.
- Documented price at review: USD 0.042 per million input tokens; output tokens free.
  This is a dated rate-card fact, not a measured package cost.
- Our choice: pin a model version for reproducibility and record the response model.
  Recheck documentation before changing transport limits or estimating future cost.

## Claude Code hooks

- [Official hooks reference](https://code.claude.com/docs/en/hooks).
- Supporting facts: command hooks receive event JSON. `SessionStart` and
  `UserPromptSubmit` can supply `hookSpecificOutput.additionalContext`. Background
  command hooks are supported; their context arrives later and separate firings are
  not automatically deduplicated.
- Our choice: capture queues locally and an explicit worker handles remote work.
  Host integration is qualified separately from JSON fixture tests. This reference
  does not establish equivalent hooks in Codex, Pi, Hermes or other harnesses.

## Claims deliberately not made

Provider examples and marketing comparisons are not Jev Wiki benchmarks. This
package has not earned a claim of superior semantic recall, zero hallucinations,
autonomous truth maintenance or lower end-to-end cost merely by using JEV. Those
claims require live, versioned, corpus-specific measurements.
