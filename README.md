# Facial Emotion Recognition - Hybrid Feature Fusion

## Setup (once)  -  use Python 3.10 or 3.11
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Put your images in `dataset/<emotion>/` (e.g. `dataset/happy/img1.jpg`, `dataset/sad/...`).

## Run in this order
| Step | Command | Output |
|---|---|---|
| 1 | `python 01_check_dataset.py` | HEIC->JPG, dataset audit, `results/dataset_issues.csv` |
| 2 | `python 02_preprocess_split.py` | face crops 224x224, `processed/train|val|test` |
| 3 | `python 03_train_custom_cnn.py` | `models/custom_cnn.keras` |
| 4 | `python 04_train_transfer_learning.py` | `models/transfer_learning.keras` |
| 5 | `python 05_train_landmark_model.py` | `models/landmark_mlp.keras` |
| 6 | `python 06_compare_individual.py` | individual comparison table + plot |
| 7 | `python 07_train_fusion.py` | CNN+TL+Landmarks fusion model |
| 8 | `python 08_ablation_study.py` | 2-branch fusion models |
| 9 | `python 09_final_evaluation.py` | final table, significance test, confusions |
| 10 | `python 10_predict.py --image photo.jpg` | emotion + confidence from every model |

All metrics, confusion matrices and training curves are saved in `results/`.
Settings (epochs, image size, split, etc.) are in `config.py`.
