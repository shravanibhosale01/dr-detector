# RetinaScan — Diabetic Retinopathy Detection System

A Flask + TensorFlow/MobileNetV2 web app that classifies retinal fundus images
into 5 diabetic retinopathy severity levels, with confidence scores, risk
classification, PDF reports, prediction history, education, and a nearby
eye-specialist locator.

## Project Structure
```
dr-detector/
├── app.py                  # Flask routes
├── config.py                # App config, class labels/info
├── inference.py              # Model loading + prediction
├── models_db.py              # SQLAlchemy history model
├── report_generator.py       # PDF report builder
├── requirements.txt
├── training/
│   └── train_model.py        # MobileNetV2 training script
├── model/                    # trained .h5 model goes here
├── templates/                # Jinja2 HTML pages
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/               # uploaded images stored here
├── instance/                 # SQLite DB
└── reports/                  # generated PDFs
```

## Setup

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model
You need the **APTOS 2019 Blindness Detection** dataset (free on Kaggle):
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Download and arrange it as:
```
training/data/
├── train.csv
└── train_images/
    ├── 000c1434d8d7.png
    └── ...
```

Then train (a GPU is strongly recommended — Google Colab works well and is free):
```bash
cd training
python train_model.py --data_dir ./data --epochs 20
```
This saves the trained model to `model/dr_mobilenetv2.h5`. Expect ~75-85%
validation accuracy with 15-20 epochs of fine-tuning; more epochs + data
augmentation will improve it further.

**No GPU/time to train?** As a placeholder to test the app end-to-end, you can
swap in any MobileNetV2-based 5-class classifier `.h5` file at that path — the
app only requires the same input shape (224x224x3) and output shape (5,).

### 3. Configure environment variables (optional)
```bash
export SECRET_KEY="something-random"
export GOOGLE_MAPS_API_KEY="your-key-here"   # for the hospital locator
```
Get a free Maps key at https://console.cloud.google.com/google/maps-apis
(enable "Maps JavaScript API" + "Places API").

### 4. Run the app
```bash
python app.py
```
Visit http://localhost:5000

## Features
- **Detection**: upload a retinal image, get a 5-class severity prediction with confidence score
- **Risk classification**: Low / Medium / High, mapped from severity class
- **PDF reports**: downloadable report per prediction (image + scores + recommendation)
- **History**: all past predictions stored in SQLite, browsable in-app
- **Education**: info on each DR stage and prevention
- **Hospital locator**: Google Maps-based nearby ophthalmologist search using the browser's geolocation

## Notes
- This is a screening aid, not a diagnostic tool — the UI includes a disclaimer accordingly.
- Swap MobileNetV2 for EfficientNet or a custom CNN in `training/train_model.py` /
  `inference.py` if you want to experiment with other architectures.
- For production, move `SECRET_KEY` and `GOOGLE_MAPS_API_KEY` out of shell exports
  and into a proper `.env` / secrets manager, and restrict the Maps API key to your domain.
