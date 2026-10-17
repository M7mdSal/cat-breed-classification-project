# Cat Breed Classification 🐱

A multi-class cat breed image classifier using **TensorFlow/Keras, MobileNetV2 transfer learning, and Streamlit**. Designed for CPU training on a modest laptop. **Dataset and pretrained model are not included**; MobileNetV2 ImageNet weights download automatically on first training run (internet required).

## Supported breeds

Example folders: `Persian`, `Siamese`, `Maine_Coon`, `Bengal`, `Ragdoll`, `British_Shorthair`. The model learns **whatever breed folders you populate**; these names are examples, not pretrained breed recognition capabilities.

## 1. Install (Windows PowerShell)

Install Python 3.11 (64-bit), then from this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get a dataset

Download a licensed, labeled cat-breed dataset (for example, the Oxford-IIIT Pet dataset, using its breed labels) and organize photos as:

```
data/
  Persian/             photo1.jpg, photo2.jpg, ...
  Siamese/             photo1.jpg, photo2.jpg, ...
  Maine_Coon/          photo1.jpg, photo2.jpg, ...
  Bengal/              photo1.jpg, photo2.jpg, ...
  Ragdoll/             photo1.jpg, photo2.jpg, ...
  British_Shorthair/   photo1.jpg, photo2.jpg, ...
```

Use at least ~50–100 varied images per breed to start, ideally more. Avoid duplicates and near-duplicates across train/validation; for rigorous testing, reserve a separate held-out test set before training. **Empty folders are placeholders only**. The script discovers classes from populated folders; remove empty folders if they cause problems with your TensorFlow version.

## 3. Train

```powershell
python src/train.py --epochs 15 --batch-size 16
```

Optional fine-tuning (slower): `python src/train.py --epochs 15 --batch-size 16 --fine-tune`

Outputs: `models/cat_breeds.keras`, `models/labels.json`, `outputs/metrics.json`, `outputs/training_curves.png`. Accuracy is measured on a validation split, **not** an independent test set. Training requires images and may take a while on CPU.

## 4. Run the web app

```powershell
streamlit run app/streamlit_app.py
```

Upload a cat photo to see the predicted breed and top 3 scores. Predictions are restricted to the breeds trained. A mixed-breed or non-cat image may still receive a high score; this is a limitation of closed-set classification.

## Architecture

160x160 RGB -> training-only image augmentation -> MobileNetV2 preprocessing -> ImageNet-pretrained MobileNetV2 feature extractor -> global average pooling -> dropout -> softmax for N breeds. The first training stage freezes MobileNetV2; optional fine-tuning unfreezes its final 20 layers. Sparse categorical cross entropy optimizes the multi-class predictions.
