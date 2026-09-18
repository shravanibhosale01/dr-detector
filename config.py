import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-this-in-production")

    # Database (prediction history)
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "history.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    # Model
    MODEL_PATH = os.path.join(BASE_DIR, "model", "dr_mobilenetv2.h5")
    IMG_SIZE = 224  # MobileNetV2 default input size

    # Reports
    REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

    # Google Maps (get a free key at https://console.cloud.google.com/google/maps-apis)
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# The five standard DR severity classes (APTOS 2019 / EyePACS grading scale)
CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

CLASS_INFO = {
    "No DR": {
        "risk": "Low",
        "description": "No visible signs of diabetic retinopathy detected in the retina.",
        "recommendation": "Continue routine annual eye screening as advised by your doctor.",
    },
    "Mild": {
        "risk": "Low",
        "description": "Early stage with microaneurysms — tiny bulges in the retina's blood vessels.",
        "recommendation": "Schedule a follow-up eye exam within the next 9-12 months.",
    },
    "Moderate": {
        "risk": "Medium",
        "description": "Blood vessels feeding the retina are becoming blocked, more visible damage present.",
        "recommendation": "See an ophthalmologist within 3-6 months for closer monitoring.",
    },
    "Severe": {
        "risk": "High",
        "description": "A significant number of blood vessels are blocked, depriving areas of the retina of blood supply.",
        "recommendation": "Seek ophthalmologist evaluation within weeks — risk of progression is high.",
    },
    "Proliferative DR": {
        "risk": "High",
        "description": "The most advanced stage — new, fragile blood vessels are growing, risking severe vision loss.",
        "recommendation": "Seek immediate ophthalmologist care. This stage requires urgent treatment.",
    },
}
