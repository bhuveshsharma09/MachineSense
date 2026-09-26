# MAFAULDA audit report — Stage 1

Source: `metadata.csv` (1,951 rows) from `scripts/01_index.py`, `00_convert.py` and `02_split.py`, seed 42,
split hash `4a93579360db1058d03fe17a03427e8d9d89ae03b409d9022f0fe0b9395eebe2`.
Figures and tables: `notebooks/01_data_layer/01_index_audit_splits.ipynb`, `results/figures/01_*.png`.
Quality flags are for auditing only and are never model inputs. No file was removed or relabelled.

## 1. Inventory

| Check | Result |
| --- | --- |
| Files | 1,951; every class count equals CLAUDE.md §3 |
| Nesting | not nested; data root is `mafaulda/` at the repo root |
| Shape | all 250,000 × 8 |
| NaN / read errors | none |
| `.npy` conversion | all 1,951 verified (float32, 250,000 × 8, finite, bit-identical to the CSV on a 5-file spot check); 15.6 GB |

## 2. Shaft speed

- Measured speed: 11.78–60.73 Hz.
- Measured/nominal: median 0.9767, IQR 0.9752–0.9780, range 0.9688–0.9838. This confirms §3's "about 2.3 % lower".
- Winning method: tacho edges on 1,845 files, tacho FFT on 106 (94 with unusable edges, 12 where FFT and comb
  confirmed each other and edges did not).
- Agreement between the three methods: median 0.010 %. 53 files differ by more than 0.2 %.
- **One file disagrees by more than 1 %:** `overhang__cage_fault__6g__16.1792`. Edges give 15.754 Hz, tacho FFT
  15.811 Hz, comb 15.602 Hz. Nothing confirms anything, so the median estimate (15.754 Hz, edges) is used.
- The accelerometer comb reads about 0.4–0.8 % low on some low-speed and overhang-ball files where both tacho
  methods agree. The comb is the least precise method and only wins when the tacho fails.
- Low-speed agreement (below about 14 Hz) is looser: the comb has fewer resolvable harmonics there.

## 3. Tacho glitches (contradicts §3 in size)

A glitch is a displaced tacho pulse: the leading edge arrives early, turning one pair of revolutions into a
short and a long interval (for example 936 + 1,134 samples against a 1,028-sample period). Speed is computed from
the clean intervals, so glitched files keep a correct speed.

| Glitched share of intervals | Files |
| --- | --- |
| any | 228 |
| > 5 % | 152 |
| > 20 % (tacho edges unusable; tacho FFT used) | 94 |

The 94 severe files: underhang ball 63 (0 g 16, 6 g 19, 20 g 17, 35 g 11), overhang outer 11, overhang ball 9,
overhang cage 6, underhang cage 5.

§3 says "about 60 files, mostly ball faults". That matches the severe underhang-ball group (63). Counting any
glitch, 228 files across eight classes are affected, including 18 imbalance and 1 horizontal misalignment.

## 4. Microphone spikes (class-correlated: leakage risk)

164 files contain bursts of 5–10 samples up to about 160 V; the median |mic| is 0.13 V. The robust z-score of
the largest sample exceeds 100 (accelerometers never exceed 51).

| Class | Masses with spikes (files) |
| --- | --- |
| imbalance | 20 g (5), 25 g (2), 30 g (37), 35 g (38) |
| underhang cage | 6 g (14), 20 g (37), 35 g (31) |

No other class has any. These look like acquisition-session artefacts rather than fault physics. Any feature
sensitive to microphone peaks (max, kurtosis, crest factor, unclipped spectrogram energy) could identify
"imbalance 30–35 g or underhang cage" from the artefact alone. **Handling needs a decision before Stage 2**
(for example robust clipping of the microphone channel, or excluding mic peak statistics).

## 5. Clipping

- **Accelerometer saturation**: 9 overhang ball 6 g files at 47–55 Hz (`overhang__ball_fault__6g__47.104`
  through `…__55.296`). The underhang accelerometers sit on their rails for up to 1,432 samples (axial),
  432 (radial) and 997 (tangential). Severity grows with speed. 8 of the 9 also have tacho glitches.
- **Microphone railing**: 8 underhang cage files (1 at 20 g, 7 at 35 g, 24.8–37.1 Hz), 10–17 samples at the
  rail. All 8 also have mic spikes.

## 6. DC offsets

- Tacho and the underhang accelerometers are near zero-mean (|DC| ≤ 0.074 V).
- The overhang accelerometers carry DC: 44 files exceed 0.2 V on at least one overhang channel, up to 2.21 V
  (axial), 1.28 V (radial) and 0.86 V (tangential). These are spread across classes (imbalance 16, underhang
  cage 6, vertical misalignment 6, underhang ball 5, others ≤ 4).
- One microphone offset: `underhang__cage_fault__35g__56.1152`, −4.53 V.
- Whether to remove DC per window is a Stage 2 representation decision. Rule 7 (global per-channel
  normalisation, no per-window z-scores) is unaffected.

## 7. Coverage and splits

- Every class covers 10–60 Hz. Overhang ball thins out above 35 Hz (4–11 recordings per 5 Hz band against about
  20 elsewhere).
- `split_closed`: within 0.7 % of 70/15/15 for every class; balanced across speed terciles (test 95/95/100).
- `loco_<class>`: 9 columns; the withheld class is never in train or val; normal is never withheld.
- `split_speed_block`: 776 test recordings (40 %). It uses **measured** speed, so 40 files with a nominal
  speed ≥ 40 Hz but measured < 40 Hz are on the training side. Overhang ball has the smallest test share (26 %).
- `split_severity`: 370 test recordings (imbalance 35 g, horizontal 2.0 mm, vertical 1.90 mm, bearing 35 g
  added mass); normal follows `split_closed`.
- `cv_fold`: 5 folds over `split_closed` train + val; per-class fold sizes differ by at most 1.

## 8. Signal exploration (notebook 02)

Source: `notebooks/01_data_layer/02_signal_exploration.ipynb`, caches from `scripts/03_signal_scan.py`, figures
`results/figures/02_*.png`. Parameters come from train recordings only; test recordings were not loaded.

### Microphone events flagged by the burst threshold (resolves section 4)
- Threshold `mic.burst_z = 18.2`: 1.25 × the largest `peakz_mic` of `split_closed` train normal recordings (14.57).
  Clean gap: the largest unflagged file is 14.57, the smallest flagged file 21.18.
- Flagged: 168 files, listed in `results/mic_burst_files.csv`: imbalance 20–35 g (83), underhang cage 6–35 g (84),
  and one overhang cage 35 g file (`overhang__cage_fault__35g__24.7808`, val, peak z 44.5).
- The flagged events look electrical rather than acoustic: single samples down to about −200 V, symmetric ringing of
  about 0.3 ms, median flagged run 7.5 samples, and no matching accelerometer response (median robust z 2.5 at the
  events, 2.5 at random times). This is an interpretation, not a proof.
- **Spot check** (visual, train + val, not an error-rate estimate): 14 of 15 random flagged files show that signature;
  the 15th (`overhang__cage_fault__35g__24.7808`) has the same shape at −7 V. There are only 6 unflagged files with
  peak z between 8 and 18.2, not 15; all 6 were inspected and none has the signature. Their events span several
  samples, reach a few volts and do not ring; two coincide with accelerometer movement at z 24–26.
- **Padding changed from 5 to 15 samples** on event-aligned evidence (rule fixed before looking: change only if the
  cleaned signal, aligned on flagged runs, stays > 1.2 × its random-position baseline beyond the patch and decays).
  With `pad = 5` the cleaned microphone was 7.8 × baseline one sample past the patch and returned within 1.2 × only
  after 15 samples, identically before and after the run. With `pad = 15` it stays within 1.15 ×, and the post-event
  RMS ratio fell from 1.32 to 1.00. Group comparisons with matched controls were not used: the most affected folders
  have only 7–11 unflagged train + val files, before any speed matching.
- Shortcut test (RandomForest on mic max, peak z, crest, kurtosis; 5-fold `cv_fold` CV on train + val): a flag means
  imbalance or underhang cage with 99.3 % precision (32 % recall). Macro-F1 is 0.628 raw and 0.638 after cleaning
  with `pad = 15` (0.620 with `pad = 5`); underhang cage F1 rises from 0.76 to 0.82. Replacing ~66 × 31 samples per
  flagged file with straight lines lowers exactly those files' crest and kurtosis, so **cleaning may leave its own
  trace**. Rule 10 keeps the ban on these features as a conservative policy; whether the Stage 2 microphone
  representation carries such a trace is unresolved.

### Speed disagreement (section 2)
- `overhang__cage_fault__6g__16.1792` is non-stationary: 15.47–16.35 Hz within 5 s (std 1.5 %), with every tacho
  interval clean. The chosen speed (tacho-edge average, 15.754 Hz) is right; the FFT peak is smeared and the comb is
  pulled low. Its order spectra are blurred by about ±3 %. Other such files would need a speed-variability column.

### Spectra
- **The ~23 kHz "resonance" is a narrowband line consistent with electrical interference**: 22,028 Hz (2–3 bins of
  6.1 Hz) with sidebands at ±55–61 Hz and ±116–122 Hz (60 Hz mains), on every accelerometer (7.3–61× the 5–20 kHz
  median) and the microphone (168×). A structural resonance would be broad and sensor-specific. This contradicts the
  "resonance" wording of CLAUDE.md §3 (now updated); its per-recording level is folder-linked (section 9).
- **Channel-group pattern (unresolved observation).** Median > 1 kHz RMS relative to train normal: underhang ball
  10.8× on over_axial and 11.3× on over_tangential; underhang outer 5.0× on over_radial; overhang ball 42× on
  under_tangential. Each bearing fault responds most on the opposite-named group, which would fit swapped column
  groups, but under_tangential also responds to most faults and overhang ball raises over_radial 10.7×. Channel
  names are unchanged.
- **Demodulation bands** (`envelope.bands_hz`): largest median spectral-kurtosis contrast (train 0 g outer race −
  train normal), bands ≥ 2 kHz (12 orders at 61 Hz), lower edge ≥ 1 kHz, each channel scored against the bearing
  position it hears best: under_radial **9–11 kHz** (overhang, +0.94), over_radial **13–17 kHz** (underhang, +4.31).
  Neither is at ~23 kHz; over_radial is outside the expected 2–10 kHz.

### `cage_fault` audit (§3; finding only, labels unchanged)
- Comb prominence in envelope-order spectra, train + val 0 g, minus the normal median, on the channel that hears
  each position:

  | Class (channel) | FTF | 2×BSF | BPFO | BPFI |
  | --- | --- | --- | --- | --- |
  | underhang cage (over_radial) | +1.06 | −0.93 | +0.03 | −1.28 |
  | overhang cage (under_radial) | −0.45 | +0.95 (p 0.010) | +0.42 | −2.40 |
  | underhang outer (over_radial) | −2.68 | −3.95 | +8.98 (p < 0.001) | +10.20 (p < 0.001) |
  | overhang outer (under_radial) | +0.69 | +1.00 | +0.47 | −1.68 |
  | underhang ball (over_radial) | −2.47 | −2.86 | −0.51 | −1.63 |
  | overhang ball (under_radial) | +3.42 (p < 0.001) | +2.51 (p < 0.001) | +1.89 | −2.20 |

- **Result: no detectable inner-race signature under this analysis; the underhang-ball positive control failed, which
  limits sensitivity.** The BPFI comb is below normal for both cage positions, while a real race defect lifts it by
  10 dB. The cage (FTF) signature is weak and not significant, so the data cannot confirm a cage fault either; at
  0 g, cage files look close to normal in the envelope.
- Limits: BPFO and BPFI sit within 0.006 orders of integer shaft orders, so those two combs cannot separate outer
  from inner race (underhang outer lifts every integer order 1–12). The ball control fails for underhang ball.

## 9. Fold-safe preprocessing choices (Stage 1c)

### Demodulation band per evaluation track
- Rule: `ml.features.select_demod_band` (docstring), parameters in `envelope.selection`, committed before it was
  applied. Applied by `scripts/04_demod_bands.py` to each track's training side only; results with the recordings
  used in `results/demod_bands.json`. Stage 2 feature code must read the band per track from that file.

  | Track | under_radial | over_radial |
  | --- | --- | --- |
  | closed and 7 LOCO folds (imbalance, both misalignments, all ball and cage folds) | 9–11 kHz (+0.94) | 13–17 kHz (+4.31) |
  | loco_underhang_outer | 9–11 kHz (+0.94) | **1–3 kHz (+0.01)** |
  | loco_overhang_outer | **23–25 kHz (+0.11)** | 13–17 kHz (+4.31) |

  Contrast (median spectral kurtosis, outer race 0 g − normal) in brackets. When the outer-race class a channel
  responds to is withheld, the remaining contrast is close to zero: the rule still returns a band, but it carries
  little information. No minimum-contrast cut was defined in advance, so none is applied.
- **The bands chosen in notebook 02 used outer-race train data**, so they must not be reused in the outer-race LOCO
  folds (`loco_underhang_outer`, `loco_overhang_outer`); those folds use their own bands above. No LOCO results
  exist yet, so nothing is contaminated so far.

### Healthy-only track (Track B)
- Option (a), **a priori**: `envelope.healthy_band_hz = 1–5 kHz` for both radial channels, fixed from the sensor
  ratings without using any recording. The anomaly detectors are trained on healthy recordings only, so a band
  chosen with fault recordings (the other tracks) would leak fault knowledge into them. Option (b), choosing
  from healthy recordings only, has no defensible target: without faults there is nothing to contrast, and the most
  impulsive healthy band says nothing about where faults appear.
- The rule: the widest band inside the ±3 dB range of **both** sensor models (MAFAULDA does not say which model is
  on which bearing), starting at the 1 kHz floor (above 12 orders at 61 Hz = 0.73 kHz) and at least 2 kHz wide.
  - 604B31 (triaxial): 0.5–5,000 Hz ±3 dB, resonance 10 kHz (spec sheet Revision B, ECN 49214).
  - 601A01: 0.27–10,000 Hz ±3 dB, resonance 16 kHz (spec sheet Revision G, ECN 14824).
- Saved as the `healthy` entry of `results/demod_bands.json` with an empty recording list.

### Bands against the sensor ratings (reporting only; bands are PROVISIONAL)
- Ratings: IMI 604B31 triaxial, ±3 dB 0.5–5,000 Hz, resonance 10 kHz (spec sheet Revision B, ECN 49214); IMI 601A01,
  ±3 dB 0.27–10,000 Hz, resonance 16 kHz (spec sheet Revision G, ECN 14824). MAFAULDA
  (www02.smt.ufrj.br/~offshore/mfs/page_01.html) lists three 601A01 units and one 604B31 triaxial unit, and
  columns 2–4 as the underhang and 5–7 as the overhang accelerometer, but **does not say which model is on which
  bearing**. Each band is therefore checked against both.

  | Band | Tracks | vs 601A01 | vs 604B31 |
  | --- | --- | --- | --- |
  | under_radial 9–11 kHz | closed + 7 LOCO | partly outside | outside, covers its 10 kHz resonance |
  | over_radial 13–17 kHz | closed + 7 LOCO | outside, covers its 16 kHz resonance | outside |
  | over_radial 1–3 kHz | loco_underhang_outer | inside | inside |
  | under_radial 23–25 kHz | loco_overhang_outer | outside | outside |
  | both channels 1–5 kHz | healthy | inside | inside |

- Comb contrast, **train only** (0 g bearings vs normal), each fault's own comb (outer → BPFO, ball → 2×BSF,
  cage → FTF), median minus normal median in dB (one-sided Mann–Whitney p):

  | Channel | Class | Selected band | In range, 1–5 kHz |
  | --- | --- | --- | --- |
  | over_radial | underhang outer (BPFO) | **+8.99 (p < 0.001)** | +3.56 (p < 0.001) |
  | over_radial | underhang cage (FTF) | +1.01 (0.38) | −1.10 (0.59) |
  | over_radial | underhang ball (2×BSF) | −3.84 (1.0) | −1.15 (0.89) |
  | under_radial | overhang ball (2×BSF) | **+2.71 (p < 0.001)** | −0.55 (0.72) |
  | under_radial | overhang outer (BPFO) | +0.78 (0.033) | −4.55 (0.99) |
  | under_radial | overhang cage (FTF) | −0.16 (0.83) | +0.08 (0.72) |
  | under_radial | underhang ball (2×BSF) | +0.72 (0.064) | **+2.88 (p < 0.001)** |
  | under_radial | underhang outer (BPFO) | +1.78 (0.005) | −4.44 (0.97) |

  The selected bands give the stronger evidence for four of the five significant results; the in-range band gives
  the only significant underhang-ball result, on the underhang-named channel. **Not switched**; `envelope.bands_provisional`
  is true until a decision.

### Speed variability within a recording (audit columns)
- `ml.preprocess.window_speeds`: speed per 1 s window (0.5 s hop, 9 windows) from the clean tacho intervals inside
  it; a window with fewer than 3 clean intervals, or every window of a recording whose pulse train is unusable,
  falls back to the recording speed. `scripts/01b_speed_windows.py` adds `speed_min_hz`, `speed_max_hz`,
  `speed_range_pct` (range / measured speed × 100) and `speed_window_fallbacks` to `metadata.csv`. **Audit only,
  never model inputs.** Split hash unchanged (`4a93579360db…`).
- Range is NaN (unknown, not zero) for the 94 recordings where every window fell back: the severe tacho-glitch
  files (underhang ball 63, overhang outer 11, overhang ball 9, overhang cage 6, underhang cage 5).
- Median range 0.087 %. Above 1 %: 21 recordings; above 2 %: 4; above 5 %: none.

  | Class | > 1 % | > 2 % | > 5 % | unknown | n |
  | --- | --- | --- | --- | --- | --- |
  | normal | 0 | 0 | 0 | 0 | 49 |
  | imbalance | 4 | 2 | 0 | 0 | 333 |
  | horizontal misalignment | 2 | 0 | 0 | 0 | 197 |
  | vertical misalignment | 5 | 1 | 0 | 0 | 301 |
  | underhang ball | 1 | 0 | 0 | 63 | 186 |
  | underhang cage | 0 | 0 | 0 | 5 | 188 |
  | underhang outer | 0 | 0 | 0 | 0 | 184 |
  | overhang ball | 1 | 0 | 0 | 9 | 137 |
  | overhang cage | 6 | 1 | 0 | 6 | 188 |
  | overhang outer | 2 | 0 | 0 | 11 | 188 |

- Largest: `imbalance__25g__15.1552` 4.46 %, `overhang__cage_fault__6g__16.1792` 4.02 % (the speed-disagreement file),
  `imbalance__15g__21.0944` 2.64 %. These are also the files whose speed methods disagreed most (section 2).
- Whether Stage 2 uses per-window speed for the order axis is an open decision.

### Other anomalies (exploratory, time-boxed; unresolved unless stated)
- **22 kHz tone, per recording** (train + val, 1,661 recordings): line at 22,028.8 Hz (IQR 22,027.0–22,030.0), ±60 Hz
  sidebands about 2–5 % of the line. Its level is **strongly folder-linked** (Kruskal–Wallis p ≤ 6e-18 across classes on
  every channel checked): on the underhang axial channel whole folders sit at about 4–10 mV (every horizontal-
  misalignment severity, overhang ball 0 g, overhang outer 0 g, underhang ball 35 g, underhang outer 6 g) while others
  carry 130–870 mV. This pattern follows acquisition folders rather than fault physics and is consistent with
  electrical interference. It is a **shortcut risk** for any representation that reaches 22 kHz. **Proposed, not
  applied:** zero-phase IIR notches at the recording's measured line f0 and f0 ± 60, ± 120 Hz (Q ≈ 11,000, ~2 Hz wide),
  or excluding 21.8–22.3 kHz from spectrogram and log-Mel representations. The envelope bands in use already avoid it.
  Figure `results/figures/02_tone_by_class.png`.
- **Shared ~10 kHz peak** (34 train normal recordings): all three `under_*` channels have a weak peak at 9.2–10.0 kHz
  (1.4–3.1 × above the surroundings); among `over_*` only radial does (9.4 kHz). Consistent with the `under_*` group
  being the triaxial 604B31 (10 kHz resonance), not proof. **Unresolved observation.**
- **Misalignment response** (327 train misalignment recordings with ≥ 3 train normals within ±3 Hz): 1× and 2× rise on
  the overhang channels (horizontal: over_radial 1× +9.7 dB; vertical: over_tangential 2× +10.6 dB, over_radial 2×
  +5.4 dB), underhang channels within ±2.6 dB; over_axial falls for vertical misalignment (−11.5 dB at 1×).
  MAFAULDA places the underhang bearing "between the rotor and the motor", nearer the coupling, so this is again
  consistent with the channel-group pattern above, but vibration also travels through rotor and frame. **Unresolved
  observation**; channel names unchanged.

## 10. Open items and unresolved observations
Decisions for Bhuvesh:
- **Demodulation bands** are provisional: the closed-track bands lie above the rated range of at least one sensor
  model (section 9). Keep them, or switch to an in-range band?
- **Order axis**: whether Stage 2 uses per-window speed (section 9, speed variability).
- **22 kHz tone**: apply the proposed notch (or exclude 21.8–22.3 kHz) before spectral representations? (section 9)
- **Channel names**: unchanged; the group pattern is an unresolved observation.

Unresolved (documented, cause not proven):
- The physical cause of the 22 kHz line (consistent with electrical interference) and why its level is folder-linked.
- The channel-group pattern (bearing faults and misalignment respond most on the opposite-named group; shared
  ~10 kHz peak on the `under_*` channels).
- Whether the flagged microphone events are electrical (they look it), and whether cleaning leaves a trace in the
  Stage 2 microphone representation.
- `cage_fault` identity: no detectable inner-race signature under this analysis, but the underhang-ball positive
  control failed, which limits sensitivity; a cage signature is not confirmed either.
- 94 severe tacho-glitch recordings have unknown within-recording speed variability.
- In the two outer-race LOCO folds, one channel's band has near-zero contrast (section 9).
