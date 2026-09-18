from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    all_scores_json = db.Column(db.Text, nullable=False)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "filename": self.filename,
            "class": self.predicted_class,
            "risk": self.risk_level,
            "confidence": self.confidence,
            "all_scores": json.loads(self.all_scores_json),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }
