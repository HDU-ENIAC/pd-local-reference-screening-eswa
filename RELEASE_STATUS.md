# Release status and scope

This is a **publicly accessible code and result-reproduction repository**.
Repository: https://github.com/HDU-ENIAC/pd-local-reference-screening-eswa
Prepared data are deposited separately at https://doi.org/10.5281/zenodo.24540220.
No code-release DOI or GitHub release tag is asserted by this visibility change.

This is the ESWA preparation snapshot for manuscript version 165. The older
repository is retained separately. Only nominal PD pulse conditions are included
in the uploaded manifests and score files; the old alternative-pulse records are
not used in the current manuscript. The signal generator retains that optional
legacy branch, but `prepare_main.py` exports only current conditions.

Prepared waveforms, including source-validation inputs, are in the Zenodo data
archive, not in GitHub. Additional main-evaluation global-reference scores,
confidence-interval replay and the N=32 downstream replay from the local audit
are not bundled in this repository. This is not an assertion of full baseline
retraining or complete reproduction of every manuscript claim.

## Licensing and scope

- The repository owner authorized public visibility. A general reuse license
  for author-owned code and trained weights has not yet been selected. Public
  access is not a claim that the entire repository is licensed under MIT, Apache,
  CC BY or another open-source license. Applicable third-party terms remain in
  force; see THIRD_PARTY_NOTICES.md.
- The Zenodo dataset uses CC0 for source material and CC BY 4.0 for the authors'
  original contributions to the extent of their rights. That scope does not
  extend to code or weights hosted here.
- The repository supports the proposed method and score-based reproduction.
  Independent source-data generation and full baseline adaptation/training
  packages remain outside its scope.

## What has not been claimed

- Full 4200-update, three-seed retraining was not repeated in this packaging task.
- Baseline training and reference/timing perturbation inference were not repeated.
- Score-based statistical recomputation is not new experimental evidence.
- Matching the recorded numbers does not certify the entire study or guarantee
  performance on naturally occurring PDs, different sites, or future hardware.

No historical alternatives, unrelated project code, editor correspondence,
credentials, institution account settings, full raw recordings, or manuscript
revision packages are intentionally included.
