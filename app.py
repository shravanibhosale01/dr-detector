import os
import json
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort
)
from werkzeug.utils import secure_filename

from config import Config, CLASS_INFO
from models_db import db, Prediction
from inference import predict
from report_generator import generate_pdf_report

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG or JPG image.")
        return redirect(url_for("index"))

    # Save with a unique name to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        result = predict(save_path, app.config["MODEL_PATH"], app.config["IMG_SIZE"])
    except FileNotFoundError as e:
        flash(str(e))
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("index"))

    # Save to history
    record = Prediction(
        filename=unique_name,
        predicted_class=result["class"],
        risk_level=result["risk"],
        confidence=result["confidence"],
        all_scores_json=json.dumps(result["all_scores"]),
    )
    db.session.add(record)
    db.session.commit()

    return render_template(
        "result.html",
        result=result,
        image_file=unique_name,
        prediction_id=record.id,
        maps_api_key=app.config["GOOGLE_MAPS_API_KEY"],
    )


@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", records=[r.to_dict() for r in records])


@app.route("/report/<int:prediction_id>")
def download_report(prediction_id):
    record = Prediction.query.get_or_404(prediction_id)
    result = {
        "class": record.predicted_class,
        "risk": record.risk_level,
        "confidence": record.confidence,
        "description": CLASS_INFO[record.predicted_class]["description"],
        "recommendation": CLASS_INFO[record.predicted_class]["recommendation"],
        "all_scores": json.loads(record.all_scores_json),
    }
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], record.filename)
    output_path = os.path.join(app.config["REPORT_FOLDER"], f"report_{prediction_id}.pdf")

    generate_pdf_report(result, image_path, output_path)
    return send_file(output_path, as_attachment=True,
                      download_name=f"DR_Report_{prediction_id}.pdf")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/hospitals")
def hospitals():
    return render_template("hospitals.html", maps_api_key=app.config["GOOGLE_MAPS_API_KEY"])


@app.route("/api/history")
def api_history():
    records = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return jsonify([r.to_dict() for r in records])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
