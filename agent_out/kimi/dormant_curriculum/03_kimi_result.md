# Kimi Dormant Curriculum Result

Date: 2026-07-03

Kimi inspected the recovered sources and wrote:

- `01_schema_report.md`
- `02_curriculum_design.md`

Kimi also drafted:

- `curator/recovered_corpus_builder.py`
- `training/build_recovered_curriculum.py`
- `tests/test_recovered_corpus_builder.py`

Codex reviewed and hardened the implementation:

- fixed text normalization so punctuation preserves word boundaries;
- removed semantic-edge symbols after Jeff's correction; edge meaning is now
  spelled out as English edge_type + target text;
- clamped recovered confidence values into `[0, 1]`;
- added `tests/test_recovered_curriculum.py`;
- added an explicit text-alias review budget;
- added `diary_region_self_reflection_v1` and made personal-log curriculum
  inclusion explicit.

Generated artifacts:

- Dormant corpus: `D:\Axon\datasets\recovered\dormant_state_v1`
- Curriculum: `D:\Axon\datasets\recovered\curriculum_v1`

Full dormant corpus counts:

- containers: 427001
- semantic_edges: 351978
- layout metadata/groups: 94595
- diary containers: 33

Long curriculum counts:

- field_surfacing_v1: 100000
- edge_prediction_v1: 100000
- retrieval_qa_v1: 100000
- procedure_next_step_v1: 100000
- episodic_exhale_filter_v1: 38489
- contradiction_alias_curation_v1: 43955
- diary_region_self_reflection_v1: 33

Verification:

- `python -B -m pytest -q -p no:cacheprovider tests/test_recovered_corpus_builder.py tests/test_recovered_curriculum.py`
- `python -B -m pytest -q -p no:cacheprovider`
