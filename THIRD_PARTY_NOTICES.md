# Scientific and software dependencies

The shared encoder follows the ResNet-18 stem/first-stage design (He et al.,
CVPR 2016), adapted to preserve time positions. The temporal blocks follow the
dilated-convolution design discussed by Bai et al. (arXiv:1803.01271). The
conformal order-statistic threshold follows Bates et al. (Annals of Statistics,
2023) and Angelopoulos and Bates (arXiv:2107.07511). These are scientific sources,
not claims of ownership of the underlying ideas.

The package imports PyTorch, NumPy, SciPy, pandas, and Matplotlib; their own
licenses apply. Dependency distributions are not bundled. The compact encoder
implementation was extracted/refactored from the project implementation; the
residual-block structure follows torchvision's standard BasicBlock. Before a
public release, review upstream provenance and retain any applicable notices.

Frozen baseline scores correspond to these published methods as adapted in the
paper; upstream weights and implementation files are not redistributed here:

| Score name | Scientific source | Role |
| --- | --- | --- |
| `pd_cnn` | Klein et al., Scientific Reports 15, 45079 (2025) | Adapted PD spectrogram classifier |
| `cnn_bilstm` | de Sousa et al., IEEE Access 14, 76074-76091 (2026) | Adapted HFCT pulse detector |
| `quant` | Dempster et al., Data Mining and Knowledge Discovery 38, 2377-2402 (2024) | General time-series classifier |
| `moderntcn` | Luo and Wang, ICLR (2024) | General time-series classifier |
| `mantis` | Feofanov et al., arXiv:2602.17868 (2026) | Candidate-only pretrained representation |
| `mantis_reference_full` | Same backbone, reference-extended control defined in this study | Downstream reference-aware control, not a separate published method |

The original antenna dataset and provider repository are credited in README.md.
Public site and measurement identifiers are retained solely to reconstruct
experimental group membership. The dataset's raw waveforms are not bundled.
