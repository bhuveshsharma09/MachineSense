# MachineSense — design brief: a Diagnosis dashboard page

**Task:** design a full-page **Diagnosis dashboard** for one investigated recording. On the report screen, the right-hand
diagnosis panel is replaced by a single button (e.g. "Open dashboard") that opens this page. Everything the panel shows
today must still be reachable on the dashboard; nothing may be dropped silently.

---

## 1. What MachineSense is

A research demonstration (NUS-ISS project) that diagnoses a rotating machine from **one 5-second recording of 8 sensor
channels** taken on a laboratory rotor test rig (the public MAFAULDA dataset). For each recording it answers:

1. **Is the machine healthy?**
2. **If faulty, which known fault?** One of 9 fault classes (below).
3. **Is this a fault the system has never seen?** If the models are not confident or the recording looks unfamiliar,
   the verdict is "Further inspection needed" rather than a guess.

It also says **what evidence supports or contradicts** the verdict and **what to inspect next** (cited steps from a curated
guidance library), and has an **investigation assistant** (chat) whose answers are checked against the case record.

**Users:** a *mechanic* (plain words, no scores or percentages) and a *developer / researcher* (scores, thresholds,
traceability). The app has a Mechanic / Developer switch today; the dashboard should keep both audiences.

---

## 2. Current user flow

1. **Home** — full-screen illustrative **3-D rig** (motor → coupling → shaft → inner bearing housing → rotor discs →
   outer bearing housing, with sensors). Hero text: "Understand what your machine is telling you". Button **Start
   investigation**. First visit asks for a display name (stored in the browser only; not a login).
2. **Investigation dialog** — choose a **development recording** from a dropdown (grouped by class; plus an "Examples"
   group of 12 curated states, see §6) **or upload a CSV** (headerless, 8 numeric columns, 250,000 rows = 5 s at 50 kHz).
   An upload of a sealed test recording is refused (403) with an explanation; a wrong format is refused (422).
3. **Running** — progress: "Checking recording → Analysing signals → Preparing findings" (diagnosis takes about a second
   on the server; the plain-language explanation can take longer).
4. **Report screen** (what the dashboard redesigns) — layered over the 3-D rig:
   - **Rig with markers** on parts and sensors (meaning of colours in §5); click a part or sensor → an **Inspector card**
     with its description and live charts (order spectrum, envelope spectrum for bearings, waveform).
   - **Right panel ("Diagnosis")**, ~30 % of the width, Mechanic / Developer switch — **this is what becomes the
     dashboard** (full inventory in §3).
   - **"Reading details" drawer** (bottom-left button): speed, duration, and a per-channel table "largest difference vs a
     healthy reference recording vs typical healthy variation"; clicking a row selects that sensor on the rig.
   - Top controls: Reset view, Exploded view, Show/Hide sensor names, Legend.
   - **Assistant launcher** (a silver gear with "?", bottom-right) → chat panel.
   - Top bar: brand (home), "Explore an unseen fault" link, user name; Developer view also offers "Open pipeline".
5. **Other pages:** `#/case/<id>/pipeline` (an architecture diagram of the pipeline drawn from that case's trace) and
   `#/unseen` ("Explore an unseen fault": a model that never saw a fault class is shown a recording of it, step by step).
6. **Below 1024 px wide** a stacked mobile layout is used (no 3-D).

Routes are hash-based: `#/`, `#/case/<case_id>` (reopen a stored case), `#/diagnose/<rec_id>`, plus the two pages above.
A dashboard route such as `#/case/<case_id>/dashboard` fits the existing pattern.

---

## 3. Inventory of the current right panel (must all survive)

### Mechanic view (plain words; no percentages, no raw scores)
- **Finding** with a coloured dot: "Possible bearing cage fault" / "No fault detected in this recording" / "No fault
  detected — a check disagrees; repeat the reading or inspect further" / "Further inspection needed".
- **Suggested location** — e.g. "Outer bearing housing — according to the dataset label (model prediction)", with
  buttons "Show outer bearing housing on the rig". For "further inspection" it lists *candidate* parts from the two most
  likely fault types.
- **Warnings that affect this result** (amber box), e.g. "The rotation speed was not steady during the recording."
- **Why it was flagged** — the plain-language explanation (2–3 bullets + "Uncertainty" bullets) with a badge: "template
  explanation" or "AI-written explanation of the evidence".
- **Supporting checks** — one sentence, e.g. "The healthy-baseline checks also see a difference from the healthy
  recordings; the secondary model agrees."
- **What to inspect next** — cited inspection steps (first 2 shown, "Show N more"); each ends "(source: <entry id>)" and may
  carry "AI-drafted, not yet checked against a published source" or "Citation not yet checked against the full text";
  hover shows the source title. A healthy verdict shows "No inspection steps are offered for a healthy verdict."
- **Limitations** — "Research demonstration; not validated on real machines." (+ the in-sample notice for dataset
  recordings).
- **View investigation activity** — the deterministic pipeline steps.

### Developer view (collapsible sections)
- **Model outputs** — verdict chip and title, location line, in-sample notice, warnings; CNN-6 (main model) top class and
  its confidence score (MSP); the secondary logistic regression's top class and score; "Both rank the same class first" or
  not; the top-3 class scores.
- **Decision checks** — four checks, each "value relation threshold: status · margin" e.g.
  "Familiarity (Mahalanobis distance) · 10.44 ≤ 30.04 · pass · margin −19.606 (−65.26 %)"; a "Checks disagree — recheck
  recommended" alert when applicable; narrow-margin notes; "N of 4 alternative decision rules agree"; the rule line
  "Decision rule: provisional research setting" (tooltip).
- **Signal quality** — measured shaft speed (Hz), speed variation within the recording (%), tacho glitches, microphone
  bursts removed, clipping channels, warnings.
- **Explanations** — logistic-regression feature contributions (top 5), CNN attribution (top channels and order bands,
  labelled exploratory), and the explanation's source / model / prompt version / latency / fallback reason; "Regenerate
  explanation".
- **Report** — the fixed 4-section report (see §4, `/report`), every item tagged **Measured / Model / Guidance / Missing**.
- **Investigation activity** — 7 timed steps (BV Bundle verified, SQ Signal quality, MD Models scored, DR Decision rule,
  PH Physics cursors and envelope spectrum, XP Explanations, RP Report).
- **Traceability** — recording id, case id, created time, bundle manifest sha, bundle git sha, preprocess config sha, split
  hash, schema version, diagnosis time and per-step timings.
- Footer: collapsible **Research limitations** list.

**Stale copy to drop:** the "Investigation agents · Planned — Five agents are planned; none is connected" block and the
"Investigation assistant / Ask about this recording" panel footer. The assistant is now **one connected agent** (§7).

---

## 4. Data available for a case (all numbers come from these API responses; the UI never invents a number)

| Endpoint | What it gives |
| --- | --- |
| `POST /api/diagnose` → evidence record | `signal_quality` (shaft speed Hz, per-window speeds, speed range %, stable?, training and calibration speed ranges, tacho glitches, mic bursts removed, NaN count, clipping channels, **warnings[]**); `diagnosis` (**decision**: healthy / known_fault / unknown, **fault_class**, display label, **caution**, 10-class and 6-class probabilities, top class, **scores** (MSP, Mahalanobis, autoencoder error, B1s speed-aware baseline, per-channel AE errors), **thresholds** (τ_p, τ_d, τ_e, τ_b), rule comparison A–D, secondary LR top class and probabilities); `physics` (speed, rpm, bearing defect frequencies in Hz, envelope spectrum, cursor orders); `xai` (LR feature contributions; CNN attribution); `limitations[]`; `provenance`; `timing_ms` |
| `GET /api/cases/{id}/report` | 4 sections — 01 What the readings suggest · 02 Supporting evidence · 03 Conflicting or missing evidence · 04 What to inspect next — items `{tag, text, hover?, cite?}`; plus `assessment` (the four checks with value, threshold, relation, pass/fail, margin, narrow flag, status, tone; disagree reasons; rule variants) |
| `GET /api/cases/{id}/channels` | per channel (6 accelerometer axes + mic): RMS, excess kurtosis, 1× and 2× order amplitudes, envelope lines at BPFO / BPFI / 2×BSF / FTF (radial axes), each vs a **named healthy training recording at a similar speed**: ratio, change factor, typical healthy band, direction, exceeds?; `marked` (≤ 2 channels changed beyond healthy variation), `exceeding`, reference recording and its speed |
| `GET /api/cases/{id}/signals` | per channel: waveform excerpt and order spectrum (multiples of shaft speed), cursor orders |
| `GET /api/cases/{id}/activity` | the 7 pipeline steps with measured timings |
| `GET/POST /api/cases/{id}/explanation` | plain-language explanation: finding, why_flagged[], uncertainty[], next_step; source (template / language model), badge, latency, fallback reason |
| `POST /api/cases/{id}/ask`, `GET …/questions` | assistant answers (§7) |
| `GET /api/cases/{id}/trace` | the pipeline trace (used by the pipeline page) |

---

## 5. Sensors, channels and rig parts

| Channel id | Plain name | Sensor (on the rig) |
| --- | --- | --- |
| `tacho` | Tachometer (once-per-revolution pulse → shaft speed) | s_tach |
| `under_axial`, `under_radial`, `under_tangential` | Inner bearing · axial / radial / tangential | s_in — triaxial accelerometer on the **inner bearing housing** (dataset name "underhang") |
| `over_axial`, `over_radial`, `over_tangential` | Outer bearing · axial / radial / tangential | s_out — triaxial accelerometer on the **outer bearing housing** ("overhang") |
| `mic` | Microphone | s_mic |

Recording: 5.00 s at 50 kHz, 8 channels; speeds in the development data ≈ 12–40 Hz (≈ 700–2,400 rpm).

**Parts:** Motor · Coupling · Shaft · Inner bearing housing · Rotor discs · Outer bearing housing.
**Fault class → part** (location "according to the dataset label"): imbalance → rotor discs; horizontal / vertical
misalignment → coupling and shaft; underhang ball / cage / outer race → inner bearing housing; overhang ball / cage /
outer race → outer bearing housing.

**Ten classes:** Normal, Imbalance, Horizontal misalignment, Vertical misalignment, Underhang bearing · ball fault / cage
fault / outer race, Overhang bearing · ball fault / cage fault / outer race.

**Marker meanings on the rig (keep these semantics):** violet ring = selected sensor; orange glow = model highlight
(predicted location); lime dot = measured change beyond typical healthy variation (≤ 2 sensors, never on a plain healthy
verdict); amber dashed = inspect this area / a signal-quality flag. Legend note: "Highlights suggest where to inspect; they
do not confirm a fault."

---

## 6. States to design for

**Verdicts:** Healthy (green) · Healthy — caution ("a healthy-baseline check disagrees; consider a recheck", amber) · Known
fault (orange; the fault class is the title) · Further inspection needed (amber).
**Also:** running, upload refused (403), file not accepted (422), no recording yet, explanation loading, assistant
unavailable.
**The 12 curated example recordings** (Examples group): healthy · known fault × 5 (imbalance, horizontal misalignment,
vertical misalignment, underhang bearing, overhang bearing) · healthy with caution · further inspection needed · rule
variants disagree · narrow margin · signal-quality warning · speed-range warning.

---

## 7. Investigation assistant (chat; keep the launcher)

One agent with four read-only tools (diagnose, check signal, get evidence, look up guidance). Each answer shows: the
answer text, **citation chips** (entry title on hover), an **Actions** list (tool + one-line summary) and a badge —
"Checked against the case record" / "Revised once" / "Could not answer with evidence" (then the fixed report's findings
and steps are shown). With no language model available it says so in one line and points to the report. Suggested
questions: "Why was this recording flagged?", "Which sensor changed most?", "What should I inspect next?".

---

## 8. Wording and honesty rules (non-negotiable)

- Fault location is always "**according to the dataset label**" (a model prediction), kept separate from "sensor anomaly
  evidence"; per-channel changes are *measured differences from a healthy recording*, not the fault location.
- "**Model scores are not the probability that the diagnosis is correct.**" Never show a score as 1.0000 or 0.0000 (use
  "> 0.9999").
- Dataset recordings carry "**This recording was used to build the model; the result is not evidence of accuracy.**"
- "**Research demonstration; not validated on real machines.**" must stay visible somewhere on the dashboard.
- Mechanic-facing content: plain words, no percentages, no raw scores.
- Never show model reasoning; the decision rule is labelled "provisional research setting".
- The CNN attribution map is "exploratory" and never presented as evidence of location.

---

## 9. Visual language (current app)

Light theme only. Fonts: **Switzer** (UI) and **Fragment Mono** (kickers, codes, numbers). Frosted-glass panels over the
rig, rounded 12–20 px. Colour tokens (OKLCH): ink `oklch(0.23 0.006 260)`, violet `oklch(0.54 0.21 277)` (model,
selection), orange `oklch(0.66 0.19 42)` (fault / model highlight), amber `oklch(0.80 0.14 75)` (caution, warnings), lime
`oklch(0.89 0.17 118)` (measured change), green `oklch(0.64 0.11 160)` (healthy). Report tags: Measured (grey), Model
(violet), Guidance (lime), Missing (amber). Reference screenshots of earlier screens: `SystemCode/design/app*_screenshots/`.

---

## 10. Open questions for the designer

1. Dashboard as a full page (own route) or a large overlay over the rig? How does the user get back to the rig?
2. One dashboard with a Mechanic / Developer toggle, or Mechanic content first with developer detail lower down?
3. Where do the channel comparison (today's "Reading details" drawer) and the per-sensor charts live: on the dashboard,
   on the rig, or both?
4. How the rig's markers relate to dashboard cards (e.g. clicking a sensor card highlights it on the rig).
5. Mobile (< 1024 px): the dashboard probably becomes the main mobile screen.
