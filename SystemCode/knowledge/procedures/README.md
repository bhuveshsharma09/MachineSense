# knowledge/procedures/ — post-confirmation procedures (schema "procedure-1")

Separate from `knowledge/entries/` (inspection guidance, "What to inspect next"). A procedure is maintenance work for a
fault that a technician has recorded as confirmed. The app offers one **only** when the case's latest technician event is a
confirmation **and** an entry here is `verification: reviewed`, names a reviewer and a review date, and lists that fault
class and that rig part (gate: `ml.procedures.gate`, served at `GET /api/cases/{id}/procedures`). In every other case the
app says: "No verified procedure is available."

**Status: no entries.** No reviewed source for repair or maintenance procedures on this rig is available to the project
(the relevant standards are paywalled, and the rig's manufacturer documentation is not in the repository). Nothing is
written here until such a source exists and a named reviewer has checked the entry. The AI-drafted inspection entries are
never copied or relabelled here.

```yaml
schema_version: procedure-1
id: kebab-case, equal to the file stem
version: "1.0"
title: one line
machine: the machine the entry applies to (e.g. the MAFAULDA rotor test rig)
parts: [motor | coupling | shaft | in_brg | rotor | out_brg]
fault_classes: [class10 fault labels]
prerequisites: [conditions before work starts]
safety_stop: [stop conditions; at least one]
steps: [one action per line, exactly as the source states it; no values the source does not give]
source: full citation (section / page)
source_url: link or null
licence: what may be reproduced
verification: reviewed | awaiting_source | unavailable | draft
reviewer: name (required when reviewed)
reviewed_on: YYYY-MM-DD (required when reviewed)
notes: caveats
```

Entries with `awaiting_source`, `unavailable` or `draft` are never offered. The app is advisory and read-only: it does not
control machinery, does not say a repair is complete, and does not supply torques, tolerances or thresholds that an entry
does not state.
