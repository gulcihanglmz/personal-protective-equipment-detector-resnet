
# YOLO2COCO Conversion & PPE Detection Pipeline (MMYOLO & MMDetection)

This document covers converting a YOLO-formatted dataset to COCO, visualizing it, splitting into train/val/test, and then training a Faster R-CNN + ResNet50-FPN model for PPE (Personal Protective Equipment) detection.

---

## Project Structure

```
aifs/
├── configs/
│   └── default.yaml           # Training configuration
├── data/
│   └── coco_utils.py          # COCO→PyTorch Dataset & transforms
├── engine/
│   ├── train.py               # Single-epoch training function
│   └── evaluate.py            # COCO mAP evaluation
├── models/
│   └── faster_rcnn.py         # Model creation function
├── scripts/
│   ├── run_train.py           # Training script
│   └── run_inference.py       # Inference script example
├── utils/
│   └── visualizer.py          # Visualization helpers
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## YOLO2COCO Conversion and Visualization Process (MMYOLO)

This document explains the steps to convert a YOLO dataset to COCO format and visualize it using the `browse_coco_json.py` tool.

---

## Folder Structure

```
C:/SH17-dataset/
└── root/
    ├── images/                # All image files should be placed here
    └── annotations/
        └── result.json        # The JSON file converted from YOLO → COCO
```

---

## 1. Step: Converting YOLO → COCO Format

Use MMYOLO’s `dataset_converters` tool to convert YOLO-format labels into COCO format:

### Directory:
```
C:\mmyolo\tools\dataset_converters
```

### Command:
```bash
python yolo2coco.py
```

### Output:
```
Saving converted results to C:\SH17-dataset\root\annotations/result.json ...
Process finished! Please check at C:\SH17-dataset\root\annotations .
Number of images found: 8099, converted: 8099, and skipped: 0. Total annotation count: 75994.
You can use tools/analysis_tools/browse_coco_json.py to visualize!
```
---

## 2. Step: Visualizing the COCO JSON File

Run the visualization script on your newly created COCO JSON:
### Directory:
```
C:\mmyolo\tools\analysis_tools
```

### Command:
```bash
python browse_coco_json.py \
  --data-root C:/SH17-dataset/root \
  --img-dir images \
  --ann-file annotations/result.json \
  --wait-time 2 \
  --disp-all \
  --category-names person hand
```

![1](https://github.com/user-attachments/assets/da7f607e-d8fa-41c5-94f7-d1ca070ff5ad)

### Parameter Descriptions:

| Parameter            | Description                                                |
|----------------------|------------------------------------------------------------|
| `--data-root`        | The main dataset folder (contains both images and annotations) |
| `--img-dir`          | Name of the images folder (inside `data-root`)             |
| `--ann-file`         | The COCO-format JSON file (inside `data-root`)             |
| `--wait-time`        | Time to wait between images (in seconds)                   |
| `--disp-all`         | Display all annotation types (bboxes, segmentations, etc.) |
| `--category-names`   | Only show these categories (e.g., `person`, `hand`)        |

---

## 3. Step: Splitting the COCO Dataset into Train/Val/Test Subsets

Here’s how to split a single `result.json` into train, val, and test subsets.

### Before Splitting:

```
dataset/
├── images/                    # All images here (8099 total)
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── annotations/
    └── result.json            # The unified COCO JSON for the entire dataset
```

### After Splitting:

```
dataset/
├── images/                    # Remains the same (all images in one folder)
├── annotations/
│   ├── result.json
│   └── splits/
│       ├── train.json         # 5671 images
│       ├── val.json           # 1619 images
│       └── test.json          # 809 images
```

### Directory:
```
C:\mmyolo\tools\misc
```

### Command:
```bash
python coco_split.py \
  --json C:/SH17-dataset/root/annotations/result.json \
  --out-dir C:/SH17-dataset/root/annotations/splits \
  --ratios 0.7 0.2 0.1 \
  --shuffle \
  --seed 42
```

### Split Script Parameters:

| Parameter    | Description                                                         |
|--------------|---------------------------------------------------------------------|
| `--json`     | Path to the input COCO-format JSON file                             |
| `--out-dir`  | Folder where the split JSONs (train/val/test) will be saved         |
| `--ratios`   | Ratios for the splits (e.g., 0.7 for train, 0.2 for val, 0.1 for test) |
| `--shuffle`  | Shuffle images before splitting                                     |
| `--seed`     | Random seed for reproducibility                                     |

---

## Split Operation Output

```text
loading annotations into memory...
Done (t=0.52s)
creating index...
index created!
Split info: ======
Train ratio = 0.7000000000000001, number = 5671
Val ratio = 0.20000000000000004, number = 1619
Test ratio = 0.10000000000000002, number = 809
Set the global seed: 42
shuffle dataset.
Saving json to C:\SH17-dataset\root\annotations\splits\train.json
Saving json to C:\SH17-dataset\root\annotations\splits\val.json
Saving json to C:\SH17-dataset\root\annotations\splits\test.json
All done!
```

## Model Training (Faster R-CNN + ResNet50-FPN)

### Virtual Environment & Dependencies

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Config (`configs/default.yaml`)

```yaml
DATA_DIR:    "C:/SH17-dataset/root"
TRAIN_JSON:  "train.json"
VAL_JSON:    "val.json"
NUM_CLASSES: 18        # 17 PPE classes + background
BATCH_SIZE:  1
NUM_EPOCHS:  20
LR:          0.005
MOMENTUM:    0.9
WEIGHT_DECAY:0.0005
STEP_LR_SIZE:3
STEP_LR_GAMMA:0.1
DEVICE:      "cuda"    # or "cpu"
SEED:        42
```

### Start Training

```bash
cd aifs
python -m scripts.run_train
```

- **last_model.pth**: saved each epoch  
- **best_model.pth**: saved when validation mAP improves  
- **checkpoint.pth**: optional full resume (model + optimizer + scheduler + epoch)

---

## Inference & Visualization

```bash
python scripts/run_inference.py \
  --model     best_model.pth \
  --image     path/to/sample.jpg \
  --threshold 0.5 \
  --output    results.jpg
```

Uses `utils/visualizer.draw_predictions` to overlay bboxes, labels & scores and saves the result.

---

## Monitoring & Logging

- **TensorBoard**:

  ```bash
  tensorboard --logdir runs --port 6006
  ```

  View:
  - **Loss/train**  
  - **LR**  
  - **mAP/val**

- **Console Logs**:  
  - `Using device: cuda`  
  - `[Epoch 1/20] Train loss: 2.3456`

---

## License & Credits

- **License**: MIT  
- **Source**:  
  - YOLO→COCO conversion: [MMYOLO](https://github.com/open-mmlab/mmyolo)  
  - Detection pipeline: [MMDetection](https://github.com/open-mmlab/mmdetection)  
