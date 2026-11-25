# Data access and reproduction scope

## Current status

This repository provides code, model states and archived result scores. Prepared
waveforms are deposited separately on Zenodo:
https://doi.org/10.5281/zenodo.24540220 (version 1.0).

Download `pd_screening_data_v1.zip` from that record. It contains source training
and validation inputs, main-performance and design-analysis inputs, normal
references, separate threshold-calibration windows, split metadata and array
documentation. Positive labels identify simulated injected PD pulses, not measured
PD-positive events. The GitHub repository does not duplicate this data archive.

## Original measured backgrounds

The background recordings are third-party data from Klein et al. (2023):

- Dataset collection: https://doi.org/10.6084/m9.figshare.c.6628553.v1
- Data descriptor: https://doi.org/10.1038/s41597-023-02451-1
- Provider code: https://github.com/Lukykl1/dataset_pd_vsb
- Source waveforms (CC0 1.0): https://doi.org/10.6084/m9.figshare.22751654
- Source metadata (CC0 1.0): https://doi.org/10.6084/m9.figshare.22746800

`data/main_records.csv` identifies the main-evaluation measurements. Download the
original recordings from the provider and run `prepare_main.py` as documented in
README.md. The measurements supply backgrounds; the normal excitation responses
and added PD pulses are simulated. These are not measured PD-positive windows.

The two source data items are CC0 1.0; that status remains unchanged in the
prepared dataset. The dataset authors' original contributions and documentation
are CC BY 4.0 to the extent of their applicable rights. The archive's LICENSE.txt
and SOURCES.txt explain these scopes. The dataset licenses do not license this
repository's code, model weights or separate third-party dependencies.

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

## Remaining reproduction limits in this repository

1. Independent raw-to-source/design input generation scripts are not bundled.
2. The additional main-evaluation global-reference scores, conditional interval
   replay and N=32 downstream replay from the local audit are not bundled here.
3. Reference-replacement and time-shift results are archived measurements, not
   complete rerunnable perturbation-inference experiments.
4. Full baseline adaptation/training packages and upstream weights are not bundled.

## Storage and review access

GitHub is the code-maintenance location; Zenodo is the versioned data archive.
The manuscript can cite both locations, with the Zenodo DOI identifying the
prepared data version. Neither platform substitutes for the other.

For double-anonymized review, prepare a separate snapshot without author or
organization identifiers, local paths, credentials, or Git history. A private
link alone does not anonymize landing-page metadata. The public repository and
the Zenodo record identify their creators and are not anonymous review links.
