# Tables 2 and 3: Method Comparisons

This folder contains 12 self-contained ResNet-18 training scripts. Each script fixes its method near the top and uses local ImageFolder splits and a local pretrained checkpoint.

## Scripts

| Script | Configuration |
| --- | --- |
| [train_cdr_full.py](./train_cdr_full.py) | Full CDR: KL-based regional gates, detail modulation, and paired-view gate consistency; no auxiliary classification CE. |
| [train_resnet18_basic.py](./train_resnet18_basic.py) | Single-view ResNet-18 with cross-entropy classification. |
| [train_resnet18_paired.py](./train_resnet18_paired.py) | Mean cross-entropy over two color views; controls for paired-view augmentation. |
| [train_resnet18_auxcontrol.py](./train_resnet18_auxcontrol.py) | Paired-view classification plus a layer3 auxiliary classification head with CE weight 0.2; no CDR gate. |
| [train_resnet18_se.py](./train_resnet18_se.py) | SE attention after layer3 with paired-view classification. |
| [train_resnet18_cbam.py](./train_resnet18_cbam.py) | CBAM attention after layer3 with paired-view classification. |
| [train_resnet18_eca.py](./train_resnet18_eca.py) | ECA attention after layer3 with paired-view classification. |
| [train_resnet18_mixstyle.py](./train_resnet18_mixstyle.py) | Training-time feature style mixing after layer2 with paired-view classification. |
| [train_resnet18_augmix.py](./train_resnet18_augmix.py) | Clean view plus two AugMix views with Jensen-Shannon consistency; PIL-based reimplementation. |
| [train_resnet18_randstainna.py](./train_resnet18_randstainna.py) | Official RandStainNA transform imported from a local repository; paired-view classification. |
| [train_resnet18_warn.py](./train_resnet18_warn.py) | Paper-informed WARN adaptation with two-level Haar WTConv and CBAM after layers 1-3. |
| [train_resnet18_mswa.py](./train_resnet18_mswa.py) | Paper-informed MSWA adaptation with two-level db4 frequency attention and residual 1x1 projections after all four stages. |

## Setup and execution

Follow the [root instructions](../ReadMe.md) to install dependencies and prepare class folders. Replace the five `r"......."` path placeholders in the selected script: `TRAIN_DIR`, `VAL_DIR`, `TEST_DIR`, `RESNET18_WEIGHTS_PATH`, and `SAVE_DIR`. Each path has an English comment explaining its purpose.

Pretrained weights are not bundled. Download the [official torchvision ResNet-18 ImageNet-1K checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth) and set `RESNET18_WEIGHTS_PATH` to the downloaded local file. See the [checkpoint instructions](../official%20document/readme.md) for a path example. Training does not download weights automatically.

The RandStainNA method also needs `RANDSTAIN_OFFICIAL_REPO` and `RANDSTAIN_YAML`. Use stain statistics derived from the training data; see [resources](../official%20document/readme.md).

From the repository root:

```bash
python "Table 2 and Table 3/train_resnet18_basic.py"
python "Table 2 and Table 3/train_cdr_full.py"
```

Alternatively, from this folder run `python train_resnet18_basic.py`. Set a unique output directory for every method/dataset/seed, and review the image size, batch size, epochs, and learning rates before running.

## Results

Outputs include training logs and plots, JSON metrics, per-class CSV reports, ten selected checkpoints, `last_checkpoint.pth`, and an English Excel report. Comparison workbooks use `<method>_metrics_10_best_checkpoints.xlsx`; the CDR workbook uses `metrics_10_best_checkpoints.xlsx`.

Report test performance using a checkpoint selected on validation data. The five test-selected checkpoints are for diagnostics only. Macro precision/recall/F1 are ratios in JSON/CSV metrics and percentages in Excel; accuracy is a percentage throughout.

## Implementation scope

AugMix is implemented with local PIL operations rather than the official source. WARN and MSWA are paper-informed reconstructions adapted to the shared ResNet-18 setup; they do not establish equivalence to the authors' exact implementations. Their wavelet filters are implemented in PyTorch and need no `pywt` package. RandStainNA requires external official source code, which is not included here.

The full CDR script is also included in [Tables 4 and 5](../Table%204%20and%20Table%205/readme.md). The [method diagram](../method.PNG) illustrates the full CDR model.
