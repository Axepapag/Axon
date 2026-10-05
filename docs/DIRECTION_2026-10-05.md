# Direction, 2026-10-05

Recorded from Jeff's own words in the session that started this repo. Section 1 is **Jeff's decisions**. Section 2 is
**suggestions from the assistant** (not decided). Section 3 is **open questions**. Nothing in section 2 or 3 is binding.

## 0. Context

- A catastrophic loss of the D: drive destroyed the lab work that had not been backed up (the newer CoreLab trainer
  among it). Jeff judged CoreLab "never any good anyway" and is changing direction.
- Jeff is **not a programmer or software engineer**. He wants a rock-solid trainer he can operate with a **window and
  buttons** (the more GUI the better), no Python editing.
- Sources: the old repo (`G:\My Drive\Projects\Axon_old\Axon_old`, formerly `Projects\Axon`), the older trainers in
  `G:\My Drive\New folder (2)\axon7`, and the v6 generation in `...\ashes_v6_history`. Work now happens here.
- Local hardware (checked 2026-10-05): GTX 1650, 4 GiB VRAM. The old ledger records that the official Mamba
  kernels need an Ampere GPU (compute 8.0+) and do not run on this card; training on Kaggle/Colab is intended too.

## 1. Jeff's decisions

**Substrate**
1. The substrate is exactly the **95 native characters**. Nothing after the 95: no byte transport, no emoji. Anything
   that does not fit fails closed.
2. A **frozen 1024D substrate** exists: one vector carries **64 characters** (1024 / 16). It is the law forever.
   Other widths are wanted later ("we need all the other sizes as well"), starting with 1024.
3. **No D64 rails. No packing 16D cells into anything. No trained projection. No "continuous D16 port".** A
   *mechanical* process reads exact 16D from the Heart and builds the larger vectors, and breaks a core's large output
   back into 16D to hand to the Heart.

**Heart, field, cores**
4. The **Heart is the canonical source of truth**; it guards the Shared Field and serves the 16D substrate.
5. Cores are **recurrent: a GRU first** (1024 recurrent memory), maybe Mamba later; transformers are not ruled out. The
   core holds a **mirror of the Shared Field** as served by the Heart and can always look back at it for exact recall.
   Everything inside the core may be fuzzy (that is where reasoning happens).
6. Each core has a **Soul with layers**. The intended cycle: input arrives; the GRU updates its recurrent state;
   it writes its state to the Soul's **hot layer**; it emits a deliberate, query-like output to a **feed-forward
   network (FFN)** that projects up to a much larger width and back down to 1024; the GRU updates again and writes
   to the Soul again, this time **organizing and compressing hot into colder layers** (letting noise dissipate,
   keeping valuable information); it repeats until it is ready to output.
7. A core's output (1024) is mechanically broken into 16D and given to the Heart, which writes it to a **new region
   of the Shared Field** where the cores collaborate (core 1's output, core 2's output, ...). This region does not exist yet.
8. A **round-robin consolidator** swaps every heartbeat (and a round-robin for training). The consolidator is the
   core that actually writes to the field (response draft, diary, tool calls, ...). Next heartbeat, the next core.

**Trainer**
9. A **window with buttons**: create cores, choose architecture, choose curriculum, start / pause / continue, test in
   runtime, keep checkpoints. It tracks the **curricula each core has studied**, progress, checkpoint ancestry,
   unique checkpoint names, complete resume state, tested backup and recovery. Local training plus Kaggle and Colab.
10. The trainer behaves like runtime: the **same Heart / field / Dormant / Soul path**.

**Repo**
11. Start a **new repo** and take only the good parts; leave the rest behind. The old repo is kept, not modified.

Evidence Jeff cited: the old D512 GRU scored **400/400** on distracted memory recall (it can tell CAT from OWL from
DOG), so the exact 16D substrate is judged proven.

## 2. Suggestions from the assistant (not decided)

- **Output is a choice, not a vector.** A core's readout scores the 95 characters plus an "I'm done" signal and picks
  one; that pick maps to its frozen code. Exactness comes from the choice being discrete. Because every architecture
  ends this way, the trainer can treat architecture as swappable.
- **Speak one character at a time**, each conditioned on the last, and let the mechanical process pack the result into
  64-lane vectors for the Heart. Emitting all 64 characters of a vector at once from one state makes each character
  independent of its neighbours and risks misspellings. The 1024D vector is great for *reading* (64 characters per step).
- A GRU reading 1024-float vectors has to learn 64 separate lane readers; its first layer is worth designing for that.
- **Thinking in substrate**: a core can think in exact text by writing thoughts to a region through the Heart, so they
  are exact, persistent and readable by itself and by the other cores.
- What the lost CoreLab measured (old ledger): a 1024 GRU on the recovered corpus went from perplexity ~341 to ~3.2
  (bigram baseline 16.1), spelled real words (`Jeffrey status and reconnaissance and reading the context templa`) and
  garbled after ~40 unseen characters. The ledger's own verdict: next-character prediction, not reasoning.

## 3. Open questions for Jeff

1. **Lane codes.** The 1024D substrate reuses the proven 16D codes in each of its 64 lanes. Jeff's phrase "in 1024, Y
   might equal Z" could mean a separately designed codebook instead. Confirm or correct before any core trains on it.
2. **How a core speaks:** one character at a time (suggested) or 64 at once?
3. **Dormant and outside characters.** Dormant currently preserves original records byte-exactly (including old
   memories with emoji or curly quotes), and only 95-clean text can be surfaced into the field. Keep it that way, or
   convert on import?
4. **The multi-core output region:** its name, whether it is canonical or a per-heartbeat board, and the schema bump
   (the field is `shared-field-v4`; v5 would add it). Also which v4 regions to retire (`trainer_instructions`,
   `training_responses` served the old D64 training view).
5. **Recovered memories.** The old memory databases (`brain.db`, `brain_recovered.db`, ...) sit in
   `G:\My Drive\New folder (2)\`. They were not opened. Should they be imported into Dormant with `curator/import_d00_memories.py`?
6. **Other widths** (128 / 256 / 512 / 2048 ...) and whether they follow the same 16-float lane rule.
7. **Which curricula first**, and who converts the non-95 ones (see `curricula/CURRICULA_AUDIT.md`).
8. **Identity and constitution.** The old `identity` region text (`history/old_axon/docs/AXON_IDENTITY_V1.md`) uses
   characters outside the 95 and has not been re-ratified for the new repo.

## 4. Next phases (proposed order)

1. Jeff answers section 3, items 1-4.
2. Write the new heartbeat / core registry / transaction layer (no rails), with the consolidator rotation and the
   multi-core output region (field v5).
3. Build the GRU core with mirror, Soul writes and the FFN, with exact output by choice.
4. Build the trainer engine (curriculum cursor, unique checkpoint names, full resume, ancestry manifest), then the window.
5. Add Kaggle/Colab packaging and tested backup/recovery.
