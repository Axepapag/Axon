# D64 mixer copy-alignment renewal — steps 61–120

Author: Grok 4.6 / GitHub Copilot CLI / 2026-09-06
Status: EVIDENCE. Non-serving. No promotion. Gate still failed.

Jeff authorized one FFN256 `copy_alignment` renewal from the exact
step-60 parent after the first 60-step smoke. Job completed returncode 0
and was fetched.

## Lineage

| Field | Value |
|---|---|
| Candidate | `axon-d64-mixer-4l-ffn256-h1` `r64v2-9d4df4d17517d7eb` |
| Shape | 1 head / 4 layers / FFN 256 / D64 |
| Parent bundle | `8501e20f2ae016c4010050a6905de38b2eb85cde548edbb1cd76d7eb06b4397e` |
| Job | `c726a825ecce731cc9f3e298e8a97eae1801d8e8e9bba5a1da6ebd0f075fb5a0` |
| Packet | `0a7f779c0f5bf8481b8cffadc1c4ce7f0615a5ce54f8de8aba0ad7a9bdd84158` |
| Source | `ede48f656335a6645d39b3a08c70668a21c9834a` |
| Segment | 61–120 |
| Report | `0260fbf6d9ef4ce0b505f8da075470666cc76772c760c0c6b0ce6a5c06cca705` |
| Final bundle | `2b4aa6dbadc4515a1ac2112eb963940161b1858c2f35bbbcfd3f062853584298` |
| Serving claimed | false |

Resume reproduced the parent hole: heldout copy-gate/position 1.0,
regression position 0.667.

## What moved

Train loss 3.10 → 1.42. Heldout mean loss ~0.64 → **0.311**.
Copy-gate stayed 1.0 on heldout and regression.

## What did not move

`copy_alignment` still fails for the same two reasons:

- regression position accuracy **0.667** (need 0.95)
- regression changed-source position pair rate **0.5**

Heldout position remains 1.0. EOS-gate remains 0.0 (not this stage).
Free-running typed exact remains 0. Extra steps bought cheaper loss,
not the missing dozen regression cases.

## Verdict

Do not promote. Do not treat falling loss as motor mastery. Do not
blindly spend a third copy_alignment tranche until those 12/36
regression misses are diagnosed. Width is still not the next spend.
