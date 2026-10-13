# Supporting Checkpoint and Stain Configuration

This folder contains a RandStainNA statistics YAML file and download instructions for the ResNet-18 checkpoint. Pretrained weights and the RandStainNA source repository are not included.

| Resource | Purpose |
| --- | --- |
| [resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth) | Official torchvision ResNet-18 ImageNet-1K pretrained checkpoint; download separately. |
| [CRC_LAB_randomTrue_n0.yaml](./CRC_LAB_randomTrue_n0.yaml) | RandStainNA configuration with LAB channel statistics and Reinhard color transformation settings. |

## ResNet-18 checkpoint

Download [the official ResNet-18 checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth), corresponding to torchvision's `ResNet18_Weights.IMAGENET1K_V1`, and save it locally. See the [torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) for weight details.

Set `RESNET18_WEIGHTS_PATH = r"......."` in the selected training script to the downloaded file's actual path. For example, after saving it in this folder and running from the repository root:

```python
RESNET18_WEIGHTS_PATH = r"official document/resnet18-f37072fd.pth"
```

An absolute path also works and is independent of the launch directory. The scripts instantiate torchvision ResNet-18 without downloading weights, load the local state dictionary strictly, and replace the final classifier for the dataset's class count. Download the weights before starting training. Compatible wrapped checkpoints may use `state_dict` or `model_state_dict`; `module.` prefixes are removed by the loader.

## RandStainNA configuration

Download the official RandStainNA repository separately and install its dependencies. Configure the RandStainNA script with actual local paths:

```python
RANDSTAIN_OFFICIAL_REPO = r"......."  # Repository directory containing randstainna.py.
RANDSTAIN_YAML = r"......."  # YAML containing stain statistics from the training data.
```

The supplied YAML uses LAB color space, Reinhard transformation, `random: true`, and `n_each_class: 0`. It contains means, standard deviations, and distribution fields for the L, A, and B channels. Its filename identifies a CRC configuration; training-data provenance is not documented in this package. Confirm its suitability before using it, and derive new statistics from the training split when needed for another dataset. Validation/test data must not be used to estimate training augmentation statistics.

The script imports `RandStainNA` from the configured local repository and constructs it with `std_hyper=-0.3`, `probability=1.0`, `distribution='normal'`, and `is_train=True`. These additional paths are used only by the RandStainNA method.

See the [root README](../ReadMe.md) and [comparison instructions](../Table%202%20and%20Table%203/readme.md) for training commands.
