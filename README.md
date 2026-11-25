# Local-reference neural screening of PD candidates

Code and result-reproduction snapshot for manuscript version 165, *A Local-Reference Neural
Network for Candidate Screening in Partial Discharge Monitoring*.

This snapshot derives from the version-142 implementation. It retains the model
and score-based reproduction code, with manifests and score files limited to the
nominal PD pulse conditions used in the current manuscript. It is not yet a
complete end-to-end reproduction release. See `DATA_ACCESS.md` for missing
assets, licensing scope, and the separate data archive.

**Status: publicly accessible code and result-reproduction repository.** This is the final local-moment
network, without a ridge/subspace adjustment. The code has no dependency on the
research archive. The licensing decision for author-owned code and weights is
pending; public visibility is not a blanket open-source license. See
`RELEASE_STATUS.md` and `THIRD_PARTY_NOTICES.md`.

## Prepared data on Zenodo

The prepared training, validation, main-performance and design-analysis inputs
are hosted separately at https://doi.org/10.5281/zenodo.24540220 (version 1.0).
Download `pd_screening_data_v1.zip` and extract it beside this repository.
The data are not duplicated in GitHub. Original background material remains
CC0 1.0; the dataset authors' original contributions and documentation are
CC BY 4.0 to the extent of their applicable rights, as specified in the data
archive. These licenses do not extend to this repository's code or weights.

```text
python evaluate_dataset.py --data ../pd_screening_data_v1/main --device cpu --check-archived
```

## What can be run

| Task | Entry point | Included assets |
| --- | --- | --- |
| Test architecture, STFT and calibration | `python -m unittest discover -s tests -v` | No external data needed |
| Run a synthetic API example | `python demo.py` | Artificial background, not a paper performance test |
| Recompute Tables I-IV | `python reproduce_results.py` | Frozen per-window scores for the actual comparisons |
| Recover the plotted data for Figures 4-6 | same command | Calibration scores and archived reference/timing measurements |
| Score new candidate windows | `run_inference.py` | Three final tensor-only checkpoints |
| Regenerate main-evaluation inputs | `prepare_main.py` | Public measurement IDs and original generation rules; raw data obtained separately |
| Replay proposed-model inference | `evaluate_dataset.py` | Checkpoints; prepared main data required |
| Train the proposed network | `train.py` | Initial random states and original batch schedule; prepared source data required |

The baseline score files permit independent recalculation of all tabulated
comparisons. They do **not** constitute a fresh baseline training reproduction.
Third-party baseline implementations and pretrained weights are not bundled.
Figures 5-6 are replayed from archived numerical measurements, not newly rerun
reference-replacement and timing experiments. Schematic Figures 1-3 are not
experimental measurements and are outside this code package.

## Installation

Use Python 3.11 and a separate environment. Install a PyTorch build appropriate
for the CPU or CUDA system, then install the remaining requirements:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo.py
python reproduce_results.py
python plot_results.py
```

Run these commands from this directory. `requirements.txt` gives compatible
ranges rather than a claim that every combination has been tested. Exact
environments used for the local audit are listed in `verification/environment.json`.
Most data/training scripts refuse to overwrite their output paths; choose a new `--output`
for another run. Model loading uses `weights_only=True`; only load trusted
checkpoints.

## Model and data contract

One condition supplies raw candidate windows `x` with shape `(N,512)`, normal
reference windows `refs` with shape `(8,512)`, and **separate** normal calibration
windows `calibration` with shape `(C,512)` or `(4,128,512)`. All windows must use
the same sampling rate and amplitude units. The paper uses 40 MS/s and
approximately aligned normal excitation responses. Do not use candidate events
to construct references or calibrate the threshold.

The model estimates one standard deviation from reference samples at indices
`0:32` and `448:512`, with lower bound `1e-4`, and applies that scale to all
windows. A 64-sample unit-energy Hann STFT with hop 4 produces a `32 x 128`
log-power map. The shared encoder produces `24 x 8 x 128` features. At each
location it compares the candidate with the reference mean using learned positive
channel scales. Reference variance is supplied as a separate `log(1+V)` input;
it is not used as the residual denominator. Frequency pooling and dilated
temporal blocks produce logits; their maximum at samples 32 through 447 is the
window score.

The frozen network scores the calibration windows using the same references.
The threshold is the `ceil((C+1)*(1-alpha))`-th ordered calibration score, or
infinity when that rank exceeds `C`. Admission uses strict `score > threshold`.
`alpha` is a nominal normal-admission level, **not a guarantee of the observed
test FPR**. No anomaly-type classifier or stream-level stopping rule is included.

```text
python run_inference.py --input condition.npz --checkpoint checkpoints/611013.pt --normal-budget 32 --alpha 0.05 --output outputs/my_condition.npz
```

## Main-evaluation data

Backgrounds come from the normal-labeled recordings in Klein et al., *Scientific
Data* 10, 544 (2023), DOI: https://doi.org/10.1038/s41597-023-02451-1.
The provider's repository links the dataset collection:
https://github.com/Lukykl1/dataset_pd_vsb

Dataset collection:
https://figshare.com/collections/A_data_set_of_Signals_from_an_Antenna_for_Detection_of_Partial_Discharges_in_Overhead_Insulated_Power_Line/6628553

Obtain the manually annotated waveforms from the provider, and retain the
`<idMeasurement>.npy` filenames. `data/main_records.csv` maps the 16 selected
records to those filenames. It contains public dataset identifiers, not physical
site coordinates or acquisition timestamps. No original waveforms are included
in the public candidate ZIP.

```text
python prepare_main.py --raw-dir path/to/manual_annotation --output outputs/main_data
python evaluate_dataset.py --data outputs/main_data --device cuda --check-archived
```

The generator subtracts each recording's mean, applies a fifth-order 20 kHz
high-pass filter using forward-backward filtering, and applies a recording-level
robust scale before constructing windows. This acquisition-data preparation is
distinct from the model's reference-only standard-deviation normalization.
The energy ratio for injected pulses uses the prepared background without the
simulated normal excitation response. The nominal and changed conditions share
background segments and injected pulses; the excitation response changes in the
reference, calibration, and candidate groups together.

These are measured backgrounds with **simulated** excitation responses and PD
pulses. The results are not naturally occurring PD-event detection rates.

## Training

Source data have a different construction from main evaluation. Do not train on
`outputs/main_data` or substitute a newly generated Gaussian dataset when trying
to reproduce the paper. The required `train.npz` contract is:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x` | `(13824,512)` | Unnormalized source candidates in their original order |
| `refs` | `(108,8,512)` | Reference sets for 108 source contexts |
| `context` | `(13824,)` | Integer reference-context index for each candidate |
| `y` | `(13824,)` | Binary window labels; 6912 event-containing candidates |

The first 9216 candidates use simulated normal responses on measured backgrounds;
the remaining 4608 use measured high-energy background segments. Both groups
contain paired event-free/event-containing windows. `data/training_manifest.json`
records the context order. The prepared source inputs and their validation split
are supplied in the separate Zenodo data archive, not in this code repository.
Independent raw-to-source regeneration scripts are not yet bundled; access to
prepared inputs does not establish complete raw-to-results reproducibility.

```text
python train.py --data path/to/train.npz --seed 611013 --device cuda --output outputs/train_611013
```

Repeat for seeds 611014 and 611015. The exact random, untrained states are
provided in `initial_states/` to retain the original RNG initialization after
removing historical dependencies. They are not pretrained weights. Training uses
window-level BCE, AdamW (learning rate 0.001, weight decay 0.0001), batch size 64,
gradient clipping at 5, 4200 updates, and an optimizer reset after update 3600.
Batch-normalization running statistics stay at their initial values, while
affine parameters are learned. The final checkpoint is used without target-label
selection. The 3600-update boundary is an optimizer reset, not localization
pretraining. `--steps 2` runs only a smoke test, never a paper reproduction.

The main and design-analysis records do not overlap source weight training.
Both evaluation groups were considered during architecture selection, as stated
in the manuscript; do not describe these results as an untouched external test.

## Interpreting the outputs

`reproduce_results.py` recalculates thresholds and decisions from the original
scores, then averages within parent recordings. It checks all 66 rounded values
in Tables I-IV. Key anchors are 88.77% event retention and 3.22% FPR for the
proposed method, versus 71.00% and 3.63% for the PD CNN under changed normal
responses at 5 dB, `N=32`, `alpha=0.05`.

Tables III-IV concern **final binary alarms**, not anomaly-type accuracy. The
raised-threshold comparison matches calibration false-positive counts, not test
FPR. Figure 4 reuses four cyclic nested calibration sequences; these overlap
when `N>128` and coincide at `N=512`. The result CSVs use rates in `[0,1]`, while
paper tables display percentages. Site/record IDs preserve pairing and prevent
treating repeated windows, seeds, or calibration subsets as independent records.

See `verification/` for checks actually run on this snapshot and `RELEASE_STATUS.md` for
licensing and reproduction limits. No paper, model, or result is uploaded by
any script in this package.

The plot script redraws the verified experimental trends; it does not attempt
pixel-identical reproduction of the journal's typeset figures.
