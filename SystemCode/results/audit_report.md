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

## 8. Open items

- **`cage_fault` vs inner race (§3)**: not audited yet. It needs envelope spectra at FTF (0.375×) vs BPFI
  (5.002×), which is Stage 2 work (notebook 02).
- Microphone spike handling (section 4).
- The single > 1 % speed disagreement (section 2) is kept as is; its speed error is at most about 1 %.
