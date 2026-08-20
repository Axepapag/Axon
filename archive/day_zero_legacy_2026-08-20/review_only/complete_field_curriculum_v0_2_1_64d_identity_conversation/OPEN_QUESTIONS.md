# Open Questions After v0.2.1

Stamp: ChatGPT / GPT-5 / 2026-08-18

These questions do not block inspection of the v0.2.1 package. They must be
answered before their corresponding production feature is promoted.

1. What durable `State/` schema will assign globally persistent core IDs and
   preserve retired/reconfigured roster history?
2. Who may change a core's region switches: Jeff, a trusted runtime manager,
   the rotating consolidator, offline promotion, or some governed combination?
3. How are concurrent core-private mask requests persisted and surfaced in the
   operator UI without confusing them with the current canonical global mask?
4. Which mutable regions should be append-preferred versus freely replaceable?
   Diary and structured knowledge likely need correction/tombstone provenance
   even when replacement is enabled.
5. What explicit privileged pathway handles legal redaction or corruption
   repair in immutable history/tool evidence while preserving the original
   audit record?
6. Should a core's authorized mask update take effect at the next phase or only
   the next tick? v0.2 conservatively uses next tick.
7. What compact policy representation will scale from three cores to thousands
   without materializing a dense core-by-region matrix?
8. When disjoint intervals are implemented, what maximum interval count and
   normalization rules prevent fragmented attention from becoming an implicit
   retrieval engine?
9. What frozen tasks and effect sizes are sufficient to prove council uplift
   over the best single core at equal measured compute?
10. Which locally runnable semantic retrieval baseline will surface exact
    dormant spans before any offline adapter training begins?
11. Will the first trained identity learner use a new persistent core ID or
    inherit one of the eight bootstrap core identities? Recommended default:
    create a new persistent R0 identity before training and bind its soul and
    checkpoint lineage separately.
12. What minimum completed-exchange evidence warrants a diary update versus a
    diary no-op? Recommended default: record event, meaning, lesson, open
    thread, confidence, and exact pointers only when at least one changed.

No question above authorizes a live runtime change or training run.
