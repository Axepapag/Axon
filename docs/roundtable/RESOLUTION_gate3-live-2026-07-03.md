# RESOLUTION — gate3-live-2026-07-03

## Topic
What is the simplest valid reply format an echo seat should produce?

## Participants
- `echo` (echo-seat-v1)
- `kimi` (kimi-code)
- Synthesizer (table)

## Summary
`echo` produced a single JSON action object with three fields: `type: "post"`, `text` containing the echoed message, and `stamp` identifying seat/version/date.

`kimi` affirmed the same shape but wrapped the answer in prose, included a `parse_fallback: true` flag, and used an `unknown / unknown` stamp even though a canonical stamp example appeared inside the explanatory text.

## Decision
The simplest valid reply format is a JSON object containing at minimum:

```json
{
  "type": "post",
  "text": "<the echoed or substantive reply string>",
  "stamp": "<Seat> / <version> / <YYYY-MM-DD>"
}
```

Additional envelope fields may exist (e.g., metadata), but the action payload itself needs these three fields.

## Dissenting flags / Non-conformance
- `kimi`'s action carried `"parse_fallback": true`, indicating the response did not match the expected first-class action format and had to be coerced.
- `kimi`'s stamp was `"unknown / unknown / 2026-07-03"` rather than the canonical `"Kimi / kimi-code / 2026-07-03"` shown inside the prose example.

## Open questions
- Should the protocol reject actions with `parse_fallback: true` from being considered valid, or accept them as valid-but-flagged?
- Should the `stamp` field be validated against a registered seat/version registry?

## Artifacts
- `echo` action: canonical example of the required reply shape.
- `kimi` action: agreed on shape, but produced with `parse_fallback` and an `unknown` stamp.
