# Tables 4 and 5: Controlled CDR Ablations

These six independent scripts share the ResNet-18 backbone, CDR placement after layer3, paired-view data pipeline, optimizer settings, and reporting code. The ablation switches control regional scoring, feature modulation, and gate consistency.

## Ablation settings

| Script | Regional score | Detail modulation | Consistency weight | Purpose |
| --- | --- | --- | --- | --- |
| [train_cdr_full.py](./train_cdr_full.py) | Counterfactual KL | Enabled | 0.1 | Full CDR reference. |
| [train_cdr_no_stability.py](./train_cdr_no_stability.py) | Counterfactual KL | Enabled | 0.0 | Remove gate-consistency loss. |
| [train_cdr_no_modulation.py](./train_cdr_no_modulation.py) | Counterfactual KL | Bypassed | 0.1 | Retain gate scoring/consistency while bypassing feature modulation. |
| [train_cdr_uniform_gate.py](./train_cdr_uniform_gate.py) | Fixed gate of 0.5 | Enabled | 0.1 | Remove regional importance differences; constant gates make consistency loss zero. |
| [train_cdr_energy_gate.py](./train_cdr_energy_gate.py) | Regional mean squared detail energy | Enabled | 0.1 | Replace counterfactual KL scoring with detail energy. |
| [train_cdr_energy_no_stability.py](./train_cdr_energy_no_stability.py) | Regional mean squared detail energy | Enabled | 0.0 | Combine energy scoring with removal of gate consistency. |

All scripts set `AUX_WEIGHT = 0.0`: the scoring head receives no auxiliary classification CE. The default regional grid is 4x4 and the scoring temperature is 1.0. The full model is illustrated in the [method diagram](../method.PNG).

## Setup and execution

See the [root README](../ReadMe.md) for dependencies, data organization, and checkpoint requirements. Replace the five `r"......."` placeholders in each script with the training, validation, and test split directories, local ResNet-18 checkpoint path, and a unique output directory. English comments explain every path.

Pretrained weights are not bundled. Download the [official torchvision ResNet-18 ImageNet-1K checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth) and set `RESNET18_WEIGHTS_PATH` to the downloaded local file. See the [checkpoint instructions](../official%20document/readme.md) for a path example. Training does not download weights automatically.

From the repository root:

```bash
python "Table 4 and Table 5/train_cdr_full.py"
python "Table 4 and Table 5/train_cdr_no_stability.py"
python "Table 4 and Table 5/train_cdr_energy_gate.py"
```

Alternatively, from this folder run `python train_cdr_full.py`. Keep data splits, seed, training budget, and checkpoint-selection rule consistent across ablations. Use separate output directories to avoid mixing trials. For a resumed run, retain its configuration and set `RESUME = True`.

## Outputs

Each run saves `ablation_config.json`, dataset/class metadata, `train_log.csv`, plots, `last_checkpoint.pth`, ten independently selected checkpoints, per-class reports, JSON summaries, and `metrics_10_best_checkpoints.xlsx`. The workbook includes checkpoint comparisons, per-class metrics, epoch history, validation-loss-selected AUC, gate diagnostics, and English reporting notes.

Gate diagnostics are measured using a fixed first validation batch; they are probe statistics rather than dataset-wide gate estimates. Optional `scikit-learn` enables one-vs-rest macro AUC for the validation-loss-selected checkpoint; AUC can be unavailable when class coverage is insufficient.

Use validation-selected checkpoints for formal test reporting. Test-selected checkpoints are diagnostic only because selection using test metrics creates optimistic bias. Accuracy and Excel precision/recall/F1 are percentages; JSON/CSV macro precision/recall/F1 are ratios.

The full CDR script is also provided in the [comparison folder](../Table%202%20and%20Table%203/readme.md).
