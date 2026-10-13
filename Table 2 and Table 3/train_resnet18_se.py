"""Independent ResNet-18 medical image classification comparison experiment.

ONE script per method. 3 local ImageFolder splits; 1 LOCAL ResNet18 pretrained checkpoint.
10 independently selected checkpoints (5 validation, 5 test) + Excel report.
TEST-selected weights are diagnostic ONLY, never use for unbiased paper evaluation.
"""
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape, quoteattr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

# Configure all five paths below before running this script.
# Raw strings (r"...") preserve backslashes in Windows paths.
TRAIN_DIR = r"......."  # Replace with the training ImageFolder directory (one subfolder per class).
VAL_DIR = r"......."  # Replace with the validation ImageFolder directory; keep it separate from train/test.
TEST_DIR = r"......."  # Replace with the held-out test ImageFolder directory.
RESNET18_WEIGHTS_PATH = r"......."  # Replace with the local pretrained ResNet18 .pth file; see official document/readme.md.
SAVE_DIR = r"......."  # Replace with a unique output directory for this method, dataset, and seed.
RESUME = False
SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
NUM_WORKERS = 4
BACKBONE_LR = 1e-4
NEW_MODULE_LR = 5e-4
WEIGHT_DECAY = 1e-4
USE_AMP = True
AUGMIX_JSD_WEIGHT = 12.0
# Only used when MODEL_KIND == 'randstainna'. Download its official repository
# yourself and provide a training-derived stain statistics YAML file.
RANDSTAIN_OFFICIAL_REPO = r"......."  # RandStainNA only: replace with the downloaded official repository directory.
RANDSTAIN_YAML = r"......."  # RandStainNA only: replace with a training-derived stain statistics .yaml file.
CRITERIA = {"acc": ("accuracy", "max"), "loss": ("loss", "min"),
            "f1": ("macro_f1", "max"), "recall": ("macro_recall", "max"),
            "precision": ("macro_precision", "max")}

MODEL_KIND = 'se'
MODEL_DISPLAY_NAME = 'ResNet-18 + SE'
TRAINING_DESCRIPTION = 'SE after layer3, paired-view CE; attention weights random initialized, backbone locally pretrained.'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairedImageFolder(datasets.ImageFolder):
    """Two color views with the same geometric transform."""

    def __init__(self, root, image_size):
        super().__init__(root=root, transform=None)
        self.geometry = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
        ])
        # Mild color perturbations. These are NOT a validated stain-augmentation model.
        self.photometric = make_randstain_transform() if MODEL_KIND == "randstainna" else transforms.ColorJitter(
            brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01
        )
        self.tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)  # RGB PIL Image
        img = self.geometry(img)
        view_original = self.tensor_normalize(img)
        view_color = self.tensor_normalize(self.photometric(img))
        return view_original, view_color, target


def eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def make_loaders(device):
    train_path = Path(TRAIN_DIR)
    val_path = Path(VAL_DIR)
    test_path = Path(TEST_DIR)
    for split_name, split_path in (("train", train_path), ("val", val_path), ("test", test_path)):
        if not split_path.is_dir():
            raise FileNotFoundError(f"Local {split_name} directory does not exist: {split_path}")
    if len({train_path.resolve(), val_path.resolve(), test_path.resolve()}) != 3:
        raise ValueError("TRAIN_DIR, VAL_DIR, TEST_DIR must be different directories")

    train_base = AugMixImageFolder(str(train_path), IMG_SIZE) if MODEL_KIND == "augmix" else PairedImageFolder(str(train_path), IMG_SIZE)
    val_base = datasets.ImageFolder(str(val_path), transform=eval_transform(IMG_SIZE))
    test_base = datasets.ImageFolder(str(test_path), transform=eval_transform(IMG_SIZE))
    if not train_base.classes:
        raise ValueError("Training dataset has no class folders")
    if train_base.class_to_idx != val_base.class_to_idx:
        raise ValueError("Train/val class_to_idx mismatch")
    if train_base.class_to_idx != test_base.class_to_idx:
        raise ValueError("Train/test class_to_idx mismatch")
    train_set, val_set = train_base, val_base

    n_classes = len(train_base.classes)
    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    if NUM_WORKERS:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_base, batch_size=BATCH_SIZE, shuffle=False, **loader_kw)
    stats = {"classes": train_base.classes, "class_to_idx": train_base.class_to_idx,
             "train_size": len(train_set), "val_size": len(val_set), "test_size": len(test_base)}
    return train_loader, val_loader, test_loader, n_classes, stats




# Approximate AugMix augmentation primitives following the published algorithm.
# For exact published implementation, use Google Research AugMix official code.
def _augmix_op(im, severity=3):
    op = random.choice(['autocontrast','equalize','posterize','solarize',
                        'rotate','shear_x','shear_y','translate_x','translate_y'])
    f = random.uniform(0.1, 1.0) * severity / 10.0
    w, h = im.size
    if op == 'autocontrast': return ImageOps.autocontrast(im)
    if op == 'equalize': return ImageOps.equalize(im)
    if op == 'posterize': return ImageOps.posterize(im, max(2, 8 - round(f * 5)))
    if op == 'solarize': return ImageOps.solarize(im, max(1, 256 - round(f * 220)))
    if op == 'rotate': return im.rotate((1 if random.random() > .5 else -1) * f * 50, resample=Image.Resampling.BILINEAR)
    if op == 'shear_x': return im.transform(im.size, Image.Transform.AFFINE, (1, f*.7, 0, 0, 1, 0), resample=Image.Resampling.BILINEAR)
    if op == 'shear_y': return im.transform(im.size, Image.Transform.AFFINE, (1, 0, 0, f*.7, 1, 0), resample=Image.Resampling.BILINEAR)
    if op == 'translate_x': return im.transform(im.size, Image.Transform.AFFINE, (1, 0, int(f*w/2), 0, 1, 0), resample=Image.Resampling.BILINEAR)
    return im.transform(im.size, Image.Transform.AFFINE, (1, 0, 0, 0, 1, int(f*h/2)), resample=Image.Resampling.BILINEAR)

class AugMixImageFolder(PairedImageFolder):
    def augment_mix(self, img):
        image = np.asarray(img, dtype=np.float32) / 255.0
        ws = np.random.dirichlet([1.0] * 3).astype(np.float32)
        m = float(np.random.beta(1., 1.))
        mixed = np.zeros_like(image)
        for weight in ws:
            temp = img.copy()
            for _ in range(random.randint(1, 3)):
                temp = _augmix_op(temp, 3)
            mixed += weight * (np.asarray(temp, dtype=np.float32) / 255.0)
        result = (1 - m) * image + m * mixed
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(result)
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.geometry(self.loader(path))
        return (self.tensor_normalize(img),
                self.tensor_normalize(self.augment_mix(img)),
                self.tensor_normalize(self.augment_mix(img)), label)

# Faithful RandStainNA transform must be taken from its official repository.
def make_randstain_transform():
    if not Path(RANDSTAIN_OFFICIAL_REPO).is_dir():
        raise FileNotFoundError(f'Official RandStainNA repository missing: {RANDSTAIN_OFFICIAL_REPO}')
    if not Path(RANDSTAIN_YAML).is_file():
        raise FileNotFoundError(f'Training-only stain YAML file missing: {RANDSTAIN_YAML}')
    sys.path.insert(0, RANDSTAIN_OFFICIAL_REPO)
    from randstainna import RandStainNA
    return RandStainNA(yaml_file=RANDSTAIN_YAML, std_hyper=-0.3,
                       probability=1.0, distribution='normal', is_train=True)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                 nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) +
                           self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True),
                                                   x.max(1, keepdim=True).values], 1)))
        return x * sa

class ECABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        t = int(abs((math.log2(channels) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        w = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * w

class MixStyle(nn.Module):
    """MixStyle feature style mixing at layer2; active only during training."""
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p, self.alpha = p, alpha
    def forward(self, x):
        if not self.training or x.size(0) < 2 or random.random() > self.p:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True).detach()
        sig = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt().detach()
        norm = (x - mu) / sig
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((x.size(0), 1, 1, 1)).to(x.device)
        order = torch.randperm(x.size(0), device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[order]
        sig_mix = lam * sig + (1 - lam) * sig[order]
        return norm * sig_mix + mu_mix

class ResNet18Comparison(nn.Module):
    def __init__(self, num_classes, weights_path, method):
        super().__init__()
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f'Local pretrained ResNet18 weights not found: {weights_path}')
        self.backbone = resnet18(weights=None)  # OFFLINE ONLY
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        if isinstance(state, dict):
            for key in ('state_dict', 'model_state_dict'):
                if key in state:
                    state = state[key]
                    break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.method = method
        self.attention = {'se': SEBlock, 'cbam': CBAMBlock,
                          'eca': ECABlock}.get(method, lambda x: nn.Identity())(256)
        self.mixstyle = MixStyle() if method == 'mixstyle' else nn.Identity()
        self.aux_classifier = nn.Linear(256, num_classes) if method == 'auxcontrol' else None
    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        x = b.layer2(x)
        x = self.mixstyle(x)
        x = b.layer3(x)
        aux = self.aux_classifier(F.adaptive_avg_pool2d(x, 1).flatten(1)) if self.aux_classifier is not None else None
        x = self.attention(x)
        x = b.layer4(x)
        logits = b.fc(torch.flatten(b.avgpool(x), 1))
        return (logits, aux) if aux is not None else logits

def metrics_from_confusion_matrix(conf, class_names=None):
    """Compute consistent multiclass metrics from a confusion matrix.

    accuracy is in percent (0-100); precision/recall/F1 are ratios (0-1).
    Macro metrics are averages of per-class metrics, giving each class equal weight.
    A zero denominator produces a score of 0, consistent with zero_division=0.
    """
    if conf.ndim != 2 or conf.shape[0] != conf.shape[1]:
        raise ValueError("Confusion matrix must be square")
    k = conf.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(k)]
    if len(class_names) != k:
        raise ValueError("class_names length does not match the confusion matrix")
    tp = conf.diag().to(dtype=torch.float64)
    support = conf.sum(dim=1).to(dtype=torch.float64)
    predicted = conf.sum(dim=0).to(dtype=torch.float64)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = int(conf.sum().item())
    weighted = support / max(1, total)
    correct = int(tp.sum().item())

    per_class = [
        {
            "class_index": i,
            "class_name": str(class_names[i]),
            "precision": round(precision[i].item(), 8),
            "recall": round(recall[i].item(), 8),
            "f1": round(f1[i].item(), 8),
            "support": int(support[i].item()),
            "predicted": int(predicted[i].item()),
            "true_positives": int(tp[i].item()),
        }
        for i in range(k)
    ]
    return {
        "accuracy": 100.0 * correct / max(1, total),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "weighted_precision": (precision * weighted).sum().item(),
        "weighted_recall": (recall * weighted).sum().item(),
        "weighted_f1": (f1 * weighted).sum().item(),
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
    }


def evaluate_model(model, dataloader, device, num_classes, loss_fn=None,
                   desc="Evaluating", class_names=None):
    model.eval()
    total_loss, total_count = 0.0, 0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                enabled=(USE_AMP and device.type == "cuda")):
                logits = model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if loss_fn is not None:
                    loss = loss_fn(logits, labels)
                    total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            count = torch.bincount(
                labels.cpu() * num_classes + preds.cpu(),
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            conf += count
            total_count += labels.size(0)

    result = metrics_from_confusion_matrix(conf, class_names=class_names)
    result["loss"] = total_loss / max(1, total_count)
    return result


def save_per_class_csv(per_class, output_path):
    columns = ["class_index", "class_name", "precision", "recall", "f1",
               "support", "predicted", "true_positives"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_class)


def save_csv(rows, path, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_training(log_rows, save_dir):
    if not log_rows:
        return
    epochs = [int(row["Epoch"]) for row in log_rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, [row["Train_Loss"] for row in log_rows], label="Train total loss")
    ax1.plot(epochs, [row["Val_Loss"] for row in log_rows], label="Validation CE loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["Val_Accuracy"] for row in log_rows], label="Val accuracy (%)", linestyle="--")
    ax2.set_ylabel("Validation accuracy (%)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_dir / "loss_acc_plot.png", dpi=180)
    plt.close(fig)

    # All four validation metrics, expressed as percentages in the chart.
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, scale in [
        ("Val_Accuracy", "Accuracy", 1),
        ("Val_Macro_Precision", "Macro Precision", 100),
        ("Val_Macro_Recall", "Macro Recall", 100),
        ("Val_Macro_F1", "Macro F1", 100),
    ]:
        values = [float(r[key]) * scale if r.get(key) not in (None, "") else float("nan")
                  for r in log_rows]
        ax.plot(epochs, values, label=label)
    ax.set(xlabel="Epoch", ylabel="Validation metric (%)", title="Validation Classification Metrics")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_dir / "validation_metrics_plot.png", dpi=180)
    plt.close(fig)




def _excel_col(index):
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def write_excel_report(output_path, sheets):
    """Create a real .xlsx file using Python's standard-library ZIP/XML.

    Portable on Windows: no openpyxl/xlsxwriter/artifact_tool installation required.
    Each sheet is (sheet_name, header_list, rows, column_widths).
    """
    base = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    ct = 'http://schemas.openxmlformats.org/package/2006/content-types'
    pkg = 'http://schemas.openxmlformats.org/package/2006/relationships'

    def node_value(value, row_no, col_no, header=False):
        ref = f'{_excel_col(col_no)}{row_no}'
        style = ' s="1"' if header else (' s="2"' if isinstance(value, (int, float)) and not isinstance(value, bool) else '')
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return f'<c r="{ref}"{style}/>'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'<c r="{ref}"{style}><v>{value}</v></c>'
        val = escape(str(value))
        return f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{val}</t></is></c>'

    with ZipFile(output_path, 'w', compression=ZIP_DEFLATED) as zipf:
        contents = [f'<Types xmlns="{ct}">',
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                    '<Default Extension="xml" ContentType="application/xml"/>',
                    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
                    '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
        wbsheets = []
        wbrels = [f'<Relationships xmlns="{pkg}">']
        for i, (name, headers, rows, widths) in enumerate(sheets, 1):
            if len(name) > 31:
                raise ValueError("Excel sheet name exceeds 31 chars")
            contents.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            wbsheets.append(f'<sheet name={quoteattr(name)} sheetId="{i}" r:id="rId{i}"/>')
            wbrels.append(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>')
            maxcols = len(headers)
            if any(len(row) != maxcols for row in rows):
                raise ValueError(f'Sheet {name} has a row with the wrong number of columns')
            colspec = ''.join(f'<col min="{j}" max="{j}" width="{w}" customWidth="1"/>'
                              for j, w in enumerate(widths, 1))
            if len(widths) != maxcols:
                raise ValueError('Number of column widths must equal number of headers')
            rows_xml = ['<row r="1" ht="26" customHeight="1">' +
                        ''.join(node_value(v, 1, j, header=True) for j, v in enumerate(headers, 1)) + '</row>']
            for rowno, row in enumerate(rows, 2):
                cells = ''.join(node_value(v, rowno, col) for col, v in enumerate(row, 1))
                rows_xml.append(f'<row r="{rowno}">{cells}</row>')
            last = f'{_excel_col(maxcols)}{max(1, len(rows) + 1)}'
            sheet_xml = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         f'<worksheet xmlns="{base}"><dimension ref="A1:{last}"/>'
                         f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
                         f'<sheetFormatPr defaultRowHeight="17"/><cols>{colspec}</cols>'
                         f'<sheetData>{"".join(rows_xml)}</sheetData>'
                         f'<autoFilter ref="A1:{last}"/></worksheet>')
            zipf.writestr(f'xl/worksheets/sheet{i}.xml', sheet_xml)
        contents.append('</Types>')
        zipf.writestr('[Content_Types].xml', ''.join(contents))
        zipf.writestr('_rels/.rels', f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        wbrels.append(f'<Relationship Id="rId{len(sheets)+1}" Type="{rel}/styles" Target="styles.xml"/>')
        wbrels.append('</Relationships>')
        zipf.writestr('xl/_rels/workbook.xml.rels', ''.join(wbrels))
        zipf.writestr('xl/workbook.xml',
                      f'<workbook xmlns="{base}" xmlns:r="{rel}"><bookViews><workbookView/></bookViews><sheets>{"".join(wbsheets)}</sheets></workbook>')
        # CellXf 1 is header; CellXf 2 is data number 0.0000. Clear navy header.
        styles = (f'<styleSheet xmlns="{base}">'
                  '<fonts count="2"><font><sz val="11"/><name val="Aptos"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Aptos"/></font></fonts>'
                  '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF203B5B"/><bgColor indexed="64"/></patternFill></fill></fills>'
                  '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                  '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                  '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                  '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/>'
                  '<xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>'
                  '</styleSheet>')
        zipf.writestr('xl/styles.xml', styles)


def selection_paths(save_dir):
    return {(split, tag): save_dir / f'resnet18_{MODEL_KIND}_best_{split}_{tag}.pth'
            for split in ('val', 'test') for tag in CRITERIA}


def choose_and_save_best(model, reports, best_scores, best_epochs, epoch, paths):
    """Ten independently tracked best checkpoints, ties keep earliest epoch."""
    changes = []
    for split in ('val', 'test'):
        for tag, (field, mode) in CRITERIA.items():
            key = f'{split}_{tag}'
            current = float(reports[split][field])
            previous = best_scores[key]
            better = current < previous if mode == 'min' else current > previous
            if better:
                torch.save(model.state_dict(), paths[(split, tag)])
                best_scores[key] = current
                best_epochs[key] = epoch
                changes.append((key, epoch, current))
    return changes


def _pct(num):
    return round(float(num) * 100, 4)


def make_excel_tables(summaries, per_class_rows, log_rows):
    summary_headers = ["Model", "Selection Split", "Selection Metric", "Best Epoch", "Metric Value at Selection", "Checkpoint File",
                       "Validation Loss", "Validation Accuracy(%)", "Validation Precision(%)", "Validation Recall(%)", "Validation F1(%)",
                       "Test Loss", "Test Accuracy(%)", "Test Precision(%)", "Test Recall(%)", "Test F1(%)", "Usage"]
    summary_rows = []
    for item in summaries:
        v, t = item['validation'], item['test']
        summary_rows.append([
            MODEL_DISPLAY_NAME, item['selected_from'], item['selection_metric'], item['epoch'],
            round(item['selection_value'], 8), item['weight_file'],
            round(v['loss'], 5), round(v['accuracy'], 4), _pct(v['macro_precision']),
            _pct(v['macro_recall']), _pct(v['macro_f1']),
            round(t['loss'], 5), round(t['accuracy'], 4), _pct(t['macro_precision']),
            _pct(t['macro_recall']), _pct(t['macro_f1']),
            'Diagnostic only (test-selected)' if item['selected_from'] == 'test' else 'Suitable for formal reporting (validation-selected)'
        ])
    pc_headers = ['Checkpoint', 'Selection Split', 'Evaluation Split', 'Class Index', 'Class Name', 'Precision(%)',
                  'Recall(%)', 'F1(%)', 'Support', 'Predicted Samples', 'Correct Samples']
    pc_rows = [[r['weight_file'], r['selected_from'], r['eval_split'], r['class_index'],
                r['class_name'], _pct(r['precision']), _pct(r['recall']), _pct(r['f1']),
                r['support'], r['predicted'], r['true_positives']] for r in per_class_rows]
    ep_headers = ['Epoch', 'Total Training Loss', 'Main Classification Loss', 'Auxiliary Loss', 'Gate Stability Loss',
                  'Validation Loss', 'Validation Accuracy(%)', 'Validation Precision(%)', 'Validation Recall(%)', 'Validation F1(%)',
                  'Test Loss', 'Test Accuracy(%)', 'Test Precision(%)', 'Test Recall(%)', 'Test F1(%)', 'Backbone LR']
    ep_rows = []
    for e in log_rows:
        def read(k):
            v = e.get(k, None)
            return None if v is None or v == '' else float(v)
        ep_rows.append([
            int(e['Epoch']), read('Train_Loss'), read('Train_CE'), read('Train_Aux'),
            read('Train_Stability'), read('Val_Loss'), read('Val_Accuracy'),
            None if read('Val_Macro_Precision') is None else _pct(read('Val_Macro_Precision')),
            None if read('Val_Macro_Recall') is None else _pct(read('Val_Macro_Recall')),
            None if read('Val_Macro_F1') is None else _pct(read('Val_Macro_F1')),
            read('Test_Loss'), read('Test_Accuracy'),
            None if read('Test_Macro_Precision') is None else _pct(read('Test_Macro_Precision')),
            None if read('Test_Macro_Recall') is None else _pct(read('Test_Macro_Recall')),
            None if read('Test_Macro_F1') is None else _pct(read('Test_Macro_F1')),
            read('LR')])
    readme_rows = [
        ['Main Table', f'{MODEL_DISPLAY_NAME}: 10 checkpoints with four metrics on the validation and test sets.'],
        ['Metric Units', 'Accuracy, precision, recall, and F1 are percentages (0-100); loss is dimensionless.'],
        ['Macro Definition', 'Compute precision, recall, and F1 per class, then take the unweighted mean; zero denominators yield zero.'],
        ['Accuracy', 'Accuracy = number of correct predictions / total number of samples.'],
        ['Loss Definition', 'Validation/test loss is the main-head cross-entropy loss, excluding auxiliary and stability terms.'],
        ['Best Checkpoints', 'Save separate .pth files for each split by maximizing accuracy, F1, recall, and precision, and minimizing loss.'],
        ['Same Epoch', 'Multiple criteria may select the same epoch; separate files are saved and may contain identical weights.'],
        ['Test-Set Leakage', 'Selecting the best epoch using test metrics causes information leakage and optimistic bias.'],
        ['Publication Reporting', 'Use validation-selected checkpoints and report their test metrics as the main generalization results.'],
        ['Test-Selected Checkpoints', 'These five files are for analysis and diagnostics only, not for claims of unbiased test performance.'],
        ['Split Independence', 'Check medical images for patient/lesion-level leakage; ImageFolder only enforces directory-level splits.'],
        ['Weight Loading', 'Load pretrained ResNet18 weights from RESNET18_WEIGHTS_PATH; no online download is performed.'],
        ['Training Strategy', TRAINING_DESCRIPTION],
    ]
    return [
        ('Best Checkpoints', summary_headers, summary_rows,
         [19, 16, 18, 14, 19, 45] + [20]*10 + [31]),
        ('Per Class', pc_headers, pc_rows, [45, 16, 16, 13, 22, 20, 17, 14, 17, 17, 17]),
        ('Epoch History', ep_headers, ep_rows, [10]+[17]*15),
        ('Read Me', ['Item', 'Description'], readme_rows, [24, 88]),
    ]



def main():
    set_seed(SEED)
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print('Method:', MODEL_DISPLAY_NAME)
    print('WARNING: TEST-SELECTED CHECKPOINTS ARE DIAGNOSTIC ONLY (test leakage).')

    train_loader, val_loader, test_loader, nclasses, stats = make_loaders(device)
    save_json(stats, save_dir / 'dataset_info.json')
    save_json({str(i): name for i, name in enumerate(stats['classes'])}, save_dir / 'class_indices.json')

    net = ResNet18Comparison(nclasses, RESNET18_WEIGHTS_PATH, MODEL_KIND).to(device)
    backbone_params = [p for name, p in net.named_parameters()
                       if not name.startswith('backbone.fc.') and not name.startswith('attention.')
                       and not name.startswith('aux_classifier.')]
    head_params = list(net.backbone.fc.parameters()) + list(net.attention.parameters())
    if MODEL_KIND == 'auxcontrol':
        head_params += list(net.aux_classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': NEW_MODULE_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and device.type == 'cuda')

    paths = selection_paths(save_dir)
    best_scores = {f'{split}_{tag}': float('inf') if mode == 'min' else -float('inf')
                   for split in ('val', 'test') for tag, (_, mode) in CRITERIA.items()}
    best_epochs = {k: None for k in best_scores}
    last_checkpoint = save_dir / 'last_checkpoint.pth'
    train_log = save_dir / 'train_log.csv'
    fields = ['Epoch', 'Train_Loss', 'Train_CE', 'Train_Aux', 'Train_Stability',
              'Val_Loss', 'Val_Accuracy', 'Val_Macro_Precision', 'Val_Macro_Recall', 'Val_Macro_F1',
              'Test_Loss', 'Test_Accuracy', 'Test_Macro_Precision', 'Test_Macro_Recall', 'Test_Macro_F1', 'LR']
    start_epoch, log_rows = 0, []
    config_fingerprint = {'train': str(Path(TRAIN_DIR).resolve()),
                          'val': str(Path(VAL_DIR).resolve()),
                          'test': str(Path(TEST_DIR).resolve()),
                          'weights': str(Path(RESNET18_WEIGHTS_PATH).resolve()),
                          'img_size': IMG_SIZE, 'classes': stats['classes'],
                          'method': MODEL_KIND, 'seed': SEED, 'batch': BATCH_SIZE,
                           'optimizer': 'AdamW', 'backbone_lr': BACKBONE_LR, 'new_lr': NEW_MODULE_LR}

    if RESUME:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f'RESUME=True but no checkpoint: {last_checkpoint}')
        ckpt = torch.load(last_checkpoint, map_location='cpu', weights_only=True)
        if ckpt.get('config_fingerprint') != config_fingerprint:
            raise ValueError('Dataset/weights/image-size/class/grid changed; cannot safely resume.')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        best_scores = ckpt['best_scores']
        best_epochs = ckpt['best_epochs']
        start_epoch = int(ckpt['epoch']) + 1
        if not train_log.is_file():
            raise FileNotFoundError('Cannot resume: train_log.csv is missing.')
        with train_log.open('r', encoding='utf-8-sig', newline='') as f:
            log_rows = [r for r in csv.DictReader(f) if int(r['Epoch']) <= start_epoch]
        print('Resuming from epoch', start_epoch + 1)
    else:
        if last_checkpoint.exists():
            raise FileExistsError('SAVE_DIR contains an existing last_checkpoint.pth. '
                                  'Use RESUME=True or choose a new SAVE_DIR, to avoid mixing trials.')
        save_csv([], train_log, fields)

    for epoch_idx in range(start_epoch, EPOCHS):
        net.train()
        accum = np.zeros(4, dtype=np.float64)
        seen = 0
        train_bar = tqdm(train_loader, desc=f'Train {epoch_idx+1}/{EPOCHS}', file=sys.stdout)
        for batch in train_bar:
            optimizer.zero_grad(set_to_none=True)
            if MODEL_KIND == 'augmix':
                views = [v.to(device, non_blocking=True) for v in batch[:3]]
                labels = batch[3].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
                    all_logits = net(torch.cat(views, dim=0))
                    logits_clean, logits_aug1, logits_aug2 = all_logits.chunk(3, dim=0)
                    ce = criterion(logits_clean, labels)
                    # AugMix Jensen-Shannon divergence, calculated in FP32 for stability.
                    p0, p1, p2 = [F.softmax(z.float(), dim=1) for z in
                                  (logits_clean, logits_aug1, logits_aug2)]
                    p_mix = ((p0 + p1 + p2) / 3).clamp_min(1e-8)
                    jsd = sum(F.kl_div(p_mix.log(), p, reduction='batchmean')
                              for p in (p0, p1, p2)) / 3.0
                    loss = ce + AUGMIX_JSD_WEIGHT * jsd
                aux_ce = torch.zeros_like(ce)
                stability = jsd
            else:
                view1, view2, labels = batch
                view1 = view1.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
                    if MODEL_KIND == 'basic':
                        logits = net(view1)
                        ce = criterion(logits, labels)
                        aux_ce = torch.zeros_like(ce)
                    else:
                        view2 = view2.to(device, non_blocking=True)
                        predictions = net(torch.cat([view1, view2], dim=0))
                        if MODEL_KIND == 'auxcontrol':
                            logits, aux_logits = predictions
                            aa, ab = aux_logits.chunk(2, dim=0)
                            aux_ce = 0.5 * (criterion(aa, labels) + criterion(ab, labels))
                        else:
                            logits = predictions
                            aux_ce = torch.zeros([], device=device)
                        a, b = logits.chunk(2, dim=0)
                        ce = 0.5 * (criterion(a, labels) + criterion(b, labels))
                    stability = torch.zeros_like(ce)
                    loss = ce + (0.2 * aux_ce if MODEL_KIND == 'auxcontrol' else 0.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = labels.size(0)
            seen += n
            accum += n * np.array([loss.item(), ce.item(), aux_ce.item(), stability.item()])
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        losses = accum / max(seen, 1)
        reports = {
            'val': evaluate_model(net, val_loader, device, nclasses, criterion,
                                  desc='Validation', class_names=stats['classes']),
            'test': evaluate_model(net, test_loader, device, nclasses, criterion,
                                   desc='Test (diagnostic selection)', class_names=stats['classes'])
        }
        scheduler.step(reports['val']['loss'])  # NEVER let test control learning rate
        epoch = epoch_idx + 1
        record = {'Epoch': epoch, 'Train_Loss': float(losses[0]), 'Train_CE': float(losses[1]),
                  'Train_Aux': float(losses[2]), 'Train_Stability': float(losses[3]),
                  'LR': optimizer.param_groups[0]['lr']}
        for split, title in [('val', 'Val'), ('test', 'Test')]:
            r = reports[split]
            record.update({f'{title}_Loss': r['loss'], f'{title}_Accuracy': r['accuracy'],
                           f'{title}_Macro_Precision': r['macro_precision'],
                           f'{title}_Macro_Recall': r['macro_recall'],
                           f'{title}_Macro_F1': r['macro_f1']})
        log_rows.append(record)
        with train_log.open('a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(record)
        updates = choose_and_save_best(net, reports, best_scores, best_epochs, epoch, paths)
        for tag, chosen_epoch, score in updates:
            print(f'Updated {tag}: epoch={chosen_epoch}, metric={score:.5f}')

        # One rolling checkpoint for crash recovery; the 10 best model weights are distinct.
        next_ckpt = last_checkpoint.with_suffix('.pth.tmp')
        torch.save({'epoch': epoch_idx, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'best_scores': best_scores, 'best_epochs': best_epochs,
                    'config_fingerprint': config_fingerprint}, next_ckpt)
        os.replace(next_ckpt, last_checkpoint)
        plot_training(log_rows, save_dir)
        print(f'[Epoch {epoch}] val Acc={reports["val"]["accuracy"]:.2f}% '
              f'P={reports["val"]["macro_precision"]:.4f} '
              f'R={reports["val"]["macro_recall"]:.4f} '
              f'F1={reports["val"]["macro_f1"]:.4f} '
              f'| test Acc={reports["test"]["accuracy"]:.2f}%')

    # Reload EACH of the 10 actual files and evaluate on BOTH val and test.
    summaries, per_class_rows = [], []
    for split in ('val', 'test'):
        for tag, (metric_field, _) in CRITERIA.items():
            path = paths[(split, tag)]
            if not path.is_file():
                raise FileNotFoundError(f'One of the 10 best checkpoints is missing: {path}')
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            val_result = evaluate_model(net, val_loader, device, nclasses, criterion,
                                        desc=f'Final validation [{split}_{tag}]', class_names=stats['classes'])
            test_result = evaluate_model(net, test_loader, device, nclasses, criterion,
                                         desc=f'Final test [{split}_{tag}]', class_names=stats['classes'])
            name = f'{split}_{tag}'
            summary = {'selected_from': split, 'selection_metric': tag,
                       'epoch': best_epochs[name], 'selection_value': best_scores[name],
                       'weight_file': path.name,
                       'validation': val_result, 'test': test_result}
            summaries.append(summary)
            for which, report in [('val', val_result), ('test', test_result)]:
                for row in report['per_class']:
                    per_class_rows.append({'weight_file': path.name, 'selected_from': split,
                                           'eval_split': which, **row})
            save_json(summary, save_dir / f'result_{name}.json')
            print(f'{name} E{best_epochs[name]}: '
                  f'val ACC={val_result["accuracy"]:.2f}% F1={val_result["macro_f1"]:.4f}, '
                  f'test ACC={test_result["accuracy"]:.2f}% F1={test_result["macro_f1"]:.4f}')

    # Every model -> four metrics on both splits in one Excel workbook.
    excel_file = save_dir / f'{MODEL_KIND}_metrics_10_best_checkpoints.xlsx'
    write_excel_report(excel_file, make_excel_tables(summaries, per_class_rows, log_rows))
    # Structured JSON is also useful for later experiments and audit.
    save_json(summaries, save_dir / 'all_10_checkpoint_results.json')
    save_json(best_epochs, save_dir / 'best_checkpoint_epochs.json')
    print(f'Saved 10 independent .pth files + Excel report: {excel_file}')
    print('IMPORTANT: only val-selected checkpoints provide legitimate held-out test reporting.')


if __name__ == '__main__':
    main()

