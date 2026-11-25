# Data access and reproduction scope

## Current status

This repository is private. Its URL does not grant reviewer access. No prepared
waveforms, public data archive, DOI, or anonymous review link are available from
this snapshot. Do not describe it as a public or complete data release.

## Original measured backgrounds

The background recordings are third-party data from Klein et al. (2023):

- Dataset collection: https://doi.org/10.6084/m9.figshare.c.6628553.v1
- Data descriptor: https://doi.org/10.1038/s41597-023-02451-1
- Provider code: https://github.com/Lukykl1/dataset_pd_vsb

`data/main_records.csv` identifies the main-evaluation measurements. Download the
original recordings from the provider and run `prepare_main.py` as documented in
README.md. The measurements supply backgrounds; the normal excitation responses
and added PD pulses are simulated. These are not measured PD-positive windows.

The specific waveform redistribution license still needs verification. The
provider code's MIT license and the data paper's open-access license are not
automatically treated as licenses for externally hosted waveform files.

## Included assets

- Core preprocessing, network, calibration, training and inference code.
- Three trained model states and three random initialization states.
- Public measurement IDs and training/main/design context manifests.
- Archived per-window scores and numerical reference/timing measurements for
  the conditions retained in the current manuscript.
- Scripts to recalculate the four main tables and plot the archived trends.

The score files include both methods' results and do not select individual
records based on favorable outcomes. The optional alternative PD pulse condition
from the older manuscript is excluded by condition name, not by performance.

## Assets to assemble before a complete release

1. The source training and validation inputs, including the validation split and
   record-to-measurement mapping, with independently runnable generation steps.
2. Prepared candidate, reference and calibration windows, with array units and
   pairing/record/condition identifiers, once redistribution is cleared.
3. Main-evaluation global-reference scores for the local/global comparison.
4. Conditional confidence-interval computation and the N=32 downstream analysis.
5. Rerunnable reference-replacement/timing inference, beyond archived plot data.
6. Baseline adaptation scripts, configurations and upstream version references.

## Planned storage and review access

GitHub is the code-maintenance location. A versioned data archive on Mendeley
Data or Zenodo can hold the prepared inputs and a matching code/weights snapshot.
A conventional cloud drive may serve as a download mirror, not the only durable
citation. No third-party account, draft dataset or public record has been created
as part of this upload.

For double-anonymized review, prepare a separate snapshot without author or
organization identifiers, local paths, credentials, or Git history. A private
link alone does not anonymize landing-page metadata. Public release and the
code/data licenses require author approval and the data-license check above.
