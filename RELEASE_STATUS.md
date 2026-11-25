# Release status and scope

This is a **private author-review repository**, not an announced public release.
Repository: https://github.com/HDU-ENIAC/pd-local-reference-screening-eswa
Access requires GitHub authorization; the URL alone does not grant access.
No DOI or public availability statement has been created.

This is the ESWA preparation snapshot for manuscript version 165. The older
repository is retained separately. Only nominal PD pulse conditions are included
in the uploaded manifests and score files; the old alternative-pulse records are
not used in the current manuscript. The signal generator retains that optional
legacy branch, but `prepare_main.py` exports only current conditions.

Prepared waveforms are not uploaded. Main-evaluation global-reference scores,
source-validation inputs, confidence-interval replay, and the N=32 downstream
replay entry point still need to be assembled. This is not an assertion of full
baseline retraining or complete reproduction of every manuscript claim.

## Author decisions required before publication

- Confirm that all authors/institutions approve release of the authored code and
  trained weights, then choose and add an actual software/weights license.
  No MIT, Apache, or other redistribution license is assumed here.
- Resolve the data provider's specific dataset license before distributing the
  prepared waveform companion. The paper's open-access license is not used as
  evidence of a separate dataset redistribution license.
- Decide whether to also release independently runnable source-data generation
  and third-party baseline adaptation packages. The current ZIP supports the
  proposed method and score-based result reproduction, not full baseline retraining.
- Add the confirmed public URL to any future availability statement only after
  release. Do not describe this private repository as publicly available.

## What has not been claimed

- Full 4200-update, three-seed retraining was not repeated in this packaging task.
- Baseline training and reference/timing perturbation inference were not repeated.
- Score-based statistical recomputation is not new experimental evidence.
- Matching the recorded numbers does not certify the entire study or guarantee
  performance on naturally occurring PDs, different sites, or future hardware.

No historical alternatives, unrelated project code, editor correspondence,
credentials, institution account settings, full raw recordings, or manuscript
revision packages are intentionally included.
