# knowledge/ — curated maintenance references

Source material for section 04 of the diagnostic report ("What to inspect next") and, later, for the
maintenance-guidance agent (D6, CLAUDE.md §11). Every recommendation the system makes is retrieved from
an entry here and cited by its `id`. Nothing is generated at run time: model memory is never presented as a source
(§11 rule 7).

## Provenance

All eight entries were drafted with AI assistance (Claude) from general vibration-analysis knowledge. The author reviews
each entry before it ships, and each entry records this with `drafted_with_ai: true`. An AI draft is not a source:
an entry whose content rests on domain knowledge rather than on this project's own protocol is marked
`ai_drafted` until each step has been checked against a published reference, at which point its `source`
gets the section or page and its `verification` moves to `open_verified` or `citation_unverified`.

## Why entries are short paraphrases

The authoritative sources for rotating-machinery maintenance are mostly copyrighted and paywalled
(the ISO 20816 and ISO 21940 series in particular). Entries therefore hold a short paraphrase written
for this project, plus a precise citation. The application displays the citation, never the source text.
An entry must never reproduce a passage from a paywalled standard.

## Entry schema

```yaml
id: kebab-case, unique, stable            # cited in the UI; never renamed once shipped
title: one line
source: full citation, or "internal"
source_url: resolvable link, or null
licence: what may and may not be reproduced
verification: see the table below
drafted_with_ai: true or false
applies_to: [class10 labels this entry may be offered for, or "any"]
evidence_required: [feature or field names the steps depend on]
excludes: [conditions under which this entry must NOT be offered]
steps: [inspection actions, each one line]
notes: caveats a reviewer needs
```

`applies_to` narrows candidates by the diagnosed condition. `evidence_required` is the harder gate: if the
pipeline did not produce every name listed, the entry is not offered at all. That is the mechanism that
stops the assistant recommending something this system has no evidence for.

`steps` are **inspections**, not repairs. "Check coupling alignment with a dial indicator" is in scope;
"replace the bearing" is not, because confirming that a repair fixes a problem is out of scope for the
project (CLAUDE.md §1).

## verification values

| value | meaning | may ship |
| --- | --- | --- |
| `internal` | derived from this project's own audit or protocol; no external source needed | yes |
| `ai_drafted` | steps drafted with AI assistance from general domain knowledge; not yet checked against a published source | yes, with the note "AI-drafted, not yet checked against a published source" shown |
| `open_verified` | citation checked against an openly accessible full text | yes |
| `citation_unverified` | citation believed correct but the full text was not opened; no numeric thresholds are quoted from it | yes, with the note shown |
| `needs_paywalled_source` | the entry is a stub; it carries no numbers and no clause references until the standard is read | **no** |

An entry marked `needs_paywalled_source` must not be offered by the report or the assistant. It exists so
that the gap is visible rather than silently absent.

## Feature names used in `evidence_required`

From `ml.features.physics_features`. Channels are `under_axial`, `under_radial`, `under_tangential`,
`over_axial`, `over_radial`, `over_tangential`, `mic`; bearings are `under` (between motor and rotor) and
`over` (outboard of the rotor).

| name | meaning |
| --- | --- |
| `f_<ch>_o1`, `f_<ch>_o2`, … | amplitude at 1×, 2×, … the measured shaft speed |
| `f_<ch>_ratio_2x_1x` | 2× amplitude divided by 1× amplitude |
| `f_<bearing>_ratio_axial_radial_1x` | 1× amplitude, axial divided by radial |
| `f_<ch>_rms`, `f_<ch>_peak`, `f_<ch>_crest`, `f_<ch>_kurtosis` | window statistics |
| `f_env_<ch>_ftf1`, `f_env_<ch>_bsf2`, `f_env_<ch>_bpfo1`, `f_env_<ch>_bpfi1` | envelope-spectrum amplitude at the cage, ball, outer-race and inner-race defect lines |
| `speed_hz` | measured shaft speed |
| `quality.warnings` | signal-quality warnings recorded during the audit |
| `decision.decision` | the arbitrator's verdict |

## What may never be cited as support

Per CLAUDE.md §11 rule 11, the CNN gradient highlights are exploratory: an occlusion check did not confirm
band-level importance. No entry may list them in `evidence_required`, and no recommendation may rest on them.
