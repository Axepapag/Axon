# Private Memory Ingestion Specification v0.2.1

Status: schema and process design only

Stamp: ChatGPT / GPT-5 / 2026-08-18

For the first 64D identity/conversation lane, this package includes schemas and
neutral synthetic fixtures only. It does not ingest `D:\\00`, the personal log,
recovered databases, souls, or exact private conversations. Codex must build a
separate local-only Grade A/B evidence pack after accepting the reader and
three-region delta contracts.

## Authority boundary

Exact events remain outside weights and souls. Semantic retrieval may locate a
source, but reasoning receives exact readable text with provenance. A summary,
embedding, or extracted relation never becomes stronger evidence than its
source.

## Evidence grades

- Grade A: exact user/assistant messages, successful tool results, immutable
  logs. May ground an event.
- Grade B: contemporaneous Axon-authored diary or journal. May ground a
  recorded first-person reflection with that label.
- Grade C: later summaries. Retrieval aids only.
- Grade D: extracted facts, entities, and relations. Search aids only until
  supported by A or B.
- Grade S: synthetic curriculum, stories, and personality examples. Never
  autobiography.

## Required local record

Every candidate stores:

- exact source pointer and source digest;
- source grade and capture time when known;
- exact text digest and normalized retrieval text;
- provenance and privacy classification;
- lineage/session identity;
- duplicate cluster;
- factual status: recorded, inferred, disputed, or unknown;
- allowed uses and retention state;
- split assignment by whole lineage;
- review and promotion status.

Missing hashes remain explicit `HASH_REQUIRED_LOCAL_BUILDER`; they are never
invented.

## Immutable evidence and masks

Conversation history, user input, tool results, and advisor input are
core-immutable. A trusted runtime adapter may append exact ingress material.
Cores may independently adjust authorized attention boundaries over those
regions without editing their content. Every move binds the unchanged source
digest and is reversible.

A future semantic retriever copies selected exact spans into an active readable
region with source IDs. It does not silently unmask an entire dormant store or
put an opaque embedding on the reasoning surface.

## Dataset isolation

Private material is `private_local_only`. It is deduplicated before splitting.
All variants from one event/session/lineage remain together. Public synthetic
records may copy the mechanics and schema but not names, exact phrases, hashes,
or identifying event details.

## Causal evaluation

For every positive identity or memory example, generate matched conditions:

- correct exact evidence;
- absent evidence requiring unknown;
- conflicting lower-grade summary;
- swapped event;
- corrupted pointer/hash;
- irrelevant retrieved result;
- correct/zero/swapped soul where applicable.

Promotion requires the answer to follow exact supported evidence, reject a
conflicting weak summary, and abstain when evidence is absent.

## Prohibited claims

Do not claim that:

- a soul remembers an event because its vector is nonzero;
- a semantic edge proves an event;
- synthetic dialogue is autobiographical;
- a generated answer is a successful tool result;
- training ingestion changes canonical history;
- masking deleted or summarized the underlying source.
