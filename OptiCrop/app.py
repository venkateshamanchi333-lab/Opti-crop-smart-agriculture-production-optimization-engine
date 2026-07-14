import csv
import io
import json
import os
import pickle
import sqlite3
from datetime import datetime
from flask import session
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for

from train_model import FEATURES, METRICS_PATH, MODEL_PATH, train


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "opti_crop.db")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "crop_recommendation.csv")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "opticrop-dev-secret")


CROP_INFO = {
    "rice": {
        "name": "Rice",
        "image": "static/images/rice.jpg",
        "season": "Kharif",
        "productivity": "High",
        "description": "Rice grows well in warm and humid climates.",
        "tips": ["Maintain water level.", "Use balanced fertilizer.", "Control weeds."]
    },

    "maize": {
        "name": "Maize",
        "image": "static/images/maize.jpg",
        "season": "Kharif/Rabi",
        "productivity": "High",
        "description": "Maize prefers fertile and well-drained soil.",
        "tips": ["Apply nitrogen.", "Keep field weed-free.", "Provide irrigation."]
    },

    "chickpea": {
        "name": "Chickpea",
        "image": "static/images/chickpea.jpg",
        "season": "Rabi",
        "productivity": "Medium",
        "description": "Chickpea is a drought-tolerant pulse crop.",
        "tips": ["Avoid waterlogging.", "Use disease-free seeds.", "Maintain proper spacing."]
    },

    "kidneybeans": {
        "name": "Kidney Beans",
        "image": "static/images/kidneybeans.jpg",
        "season": "Rabi",
        "productivity": "Medium",
        "description": "Kidney beans need cool weather and fertile soil.",
        "tips": ["Provide moderate irrigation.", "Improve drainage.", "Apply compost."]
    },

    "pigeonpeas": {
        "name": "Pigeon Peas",
        "image": "static/images/pigeonpeas.jpg",
        "season": "Kharif",
        "productivity": "Medium",
        "description": "Pigeon pea grows well in tropical climates.",
        "tips": ["Use quality seeds.", "Avoid excess moisture.", "Control insects."]
    },

    "mothbeans": {
        "name": "Moth Beans",
        "image": "static/images/mothbeans.jpg",
        "season": "Kharif",
        "productivity": "Medium",
        "description": "Moth beans are suitable for dry regions.",
        "tips": ["Needs little irrigation.", "Maintain spacing.", "Control weeds."]
    },

    "mungbean": {
        "name": "Mung Bean",
        "image": "static/images/mungbean.jpg",
        "season": "Summer",
        "productivity": "Medium",
        "description": "Mung bean is a short-duration pulse crop.",
        "tips": ["Harvest on time.", "Avoid overwatering.", "Use organic manure."]
    },

    "blackgram": {
        "name": "Black Gram",
        "image": "static/images/blackgram.jpg",
        "season": "Kharif",
        "productivity": "Medium",
        "description": "Black gram grows well in warm climates.",
        "tips": ["Ensure drainage.", "Apply phosphorus.", "Prevent pests."]
    },

    "lentil": {
        "name": "Lentil",
        "image": "static/images/lentil.jpg",
        "season": "Rabi",
        "productivity": "Medium",
        "description": "Lentils require cool temperatures.",
        "tips": ["Use certified seeds.", "Avoid excess irrigation.", "Control weeds."]
    },

    "pomegranate": {
        "name": "Pomegranate",
        "image": "static/images/pomegranate.jpg",
        "season": "Annual",
        "productivity": "High",
        "description": "Pomegranate thrives in dry climates.",
        "tips": ["Regular pruning.", "Drip irrigation.", "Monitor fruit borers."]
    },

    "banana": {
        "name": "Banana",
        "image": "static/images/banana.jpg",
        "season": "Annual",
        "productivity": "High",
        "description": "Banana requires fertile soil and irrigation.",
        "tips": ["High potassium.", "Regular watering.", "Support plants."]
    },

    "mango": {
        "name": "Mango",
        "image": "static/images/mango.jpg",
        "season": "Summer",
        "productivity": "High",
        "description": "Mango is a tropical fruit crop.",
        "tips": ["Annual pruning.", "Balanced fertilizer.", "Pest monitoring."]
    },

    "grapes": {
        "name": "Grapes",
        "image": "static/images/grapes.jpg",
        "season": "Winter",
        "productivity": "High",
        "description": "Grapes grow well in warm dry climates.",
        "tips": ["Proper pruning.", "Support vines.", "Disease management."]
    },

    "watermelon": {
        "name": "Watermelon",
        "image": "static/images/watermelon.jpg",
        "season": "Summer",
        "productivity": "High",
        "description": "Watermelon requires sandy soil.",
        "tips": ["Regular watering.", "Mulching.", "Pollination management."]
    },

    "muskmelon": {
        "name": "Muskmelon",
        "image": "static/images/muskmelon.jpg",
        "season": "Summer",
        "productivity": "High",
        "description": "Muskmelon grows in warm climates.",
        "tips": ["Maintain soil moisture.", "Good drainage.", "Pest control."]
    },

    "apple": {
        "name": "Apple",
        "image": "static/images/apple.jpg",
        "season": "Winter",
        "productivity": "High",
        "description": "Apple requires cool climates.",
        "tips": ["Regular pruning.", "Balanced nutrients.", "Disease management."]
    },

    "orange": {
        "name": "Orange",
        "image": "static/images/orange.jpg",
        "season": "Annual",
        "productivity": "High",
        "description": "Orange trees prefer subtropical climates.",
        "tips": ["Drip irrigation.", "Micronutrients.", "Pruning."]
    },

    "papaya": {
        "name": "Papaya",
        "image": "static/images/papaya.jpg",
        "season": "Annual",
        "productivity": "High",
        "description": "Papaya grows rapidly in tropical climates.",
        "tips": ["Well-drained soil.", "Regular irrigation.", "Disease monitoring."]
    },

    "coconut": {
        "name": "Coconut",
        "image": "static/images/coconut.jpg",
        "season": "Annual",
        "productivity": "High",
        "description": "Coconut grows in coastal tropical regions.",
        "tips": ["Regular watering.", "Organic manure.", "Pest management."]
    },

    "cotton": {
        "name": "Cotton",
        "image": "static/images/cotton.jpg",
        "season": "Kharif",
        "productivity": "High",
        "description": "Cotton requires warm weather.",
        "tips": ["Use potash fertilizer.", "Monitor bollworms.", "Avoid waterlogging."]
    },

    "jute": {
        "name": "Jute",
        "image": "static/images/jute.jpg",
        "season": "Kharif",
        "productivity": "Medium",
        "description": "Jute grows in humid regions.",
        "tips": ["Requires rainfall.", "Fertile soil.", "Harvest at proper stage."]
    },

    "coffee": {
        "name": "Coffee",
        "image": "static/images/coffee.jpg",
        "season": "Annual",
        "productivity": "High",
        "description": "Coffee grows in shaded tropical areas.",
        "tips": ["Provide shade.", "Organic manure.", "Disease monitoring."]
    }
}




def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            role TEXT DEFAULT 'guest',
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crops (
            crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name TEXT UNIQUE NOT NULL,
            crop_image_url TEXT,
            suitable_season TEXT,
            expected_productivity TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crop_growing_tips (
            tip_id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_id INTEGER NOT NULL,
            tip_title TEXT NOT NULL,
            tip_description TEXT NOT NULL,
            FOREIGN KEY (crop_id) REFERENCES crops (crop_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            recommended_crop_id INTEGER,
            nitrogen REAL,
            phosphorous REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            confidence_score REAL,
            model_used TEXT,
            predicted_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (recommended_crop_id) REFERENCES crops (crop_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crop_suitability_analyses (
            suitability_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            crop_id INTEGER,
            suitability_score REAL,
            suitability_status TEXT,
            missing_nutrients TEXT,
            soil_analysis TEXT,
            climate_compatibility TEXT,
            improvement_suggestions TEXT,
            analyzed_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES user_predictions (prediction_id),
            FOREIGN KEY (crop_id) REFERENCES crops (crop_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction_id INTEGER,
            action_type TEXT NOT NULL,
            action_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (prediction_id) REFERENCES user_predictions (prediction_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
            rating INTEGER,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    cursor.execute(
        """
        INSERT OR IGNORE INTO users (user_id, full_name, email, password_hash, role, created_at)
        VALUES (1, 'Guest Farmer', 'guest@opticrop.local', '', 'guest', ?)
        """,
        (datetime.utcnow().isoformat(),),
    )
    seed_crops(cursor)
    connection.commit()
    connection.close()


def seed_crops(cursor):
    now = datetime.utcnow().isoformat()
    for crop_key, crop in CROP_INFO.items():
        cursor.execute(
            """
            INSERT OR IGNORE INTO crops
            (crop_name, crop_image_url, suitable_season, expected_productivity, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (crop_key, crop["image"], crop["season"], crop["productivity"], crop["description"], now),
        )
        crop_id = cursor.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (crop_key,)).fetchone()["crop_id"]
        for index, tip in enumerate(crop["tips"], start=1):
            cursor.execute(
                """
                INSERT INTO crop_growing_tips (crop_id, tip_title, tip_description)
                SELECT ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM crop_growing_tips WHERE crop_id = ? AND tip_description = ?
                )
                """,
                (crop_id, f"Tip {index}", tip, crop_id, tip),
            )


def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return train()
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def load_dataset():
    return pd.read_csv(DATASET_PATH).dropna().drop_duplicates()


def read_float(source, field_name):
    value = source.get(field_name, "").strip()
    if value == "":
        raise ValueError(f"{field_name} is required")
    return float(value)


def extract_inputs(source):
    return {
        "N": read_float(source, "nitrogen"),
        "P": read_float(source, "phosphorous"),
        "K": read_float(source, "potassium"),
        "temperature": read_float(source, "temperature"),
        "humidity": read_float(source, "humidity"),
        "ph": read_float(source, "ph"),
        "rainfall": read_float(source, "rainfall"),
    }


def predict_crop(inputs):
    artifact = load_artifact()
    row = pd.DataFrame([[inputs[feature] for feature in FEATURES]], columns=FEATURES)
    encoded_prediction = artifact["model"].predict(row)[0]
    crop_key = artifact["encoder"].inverse_transform([encoded_prediction])[0]
    confidence = 0.0
    if hasattr(artifact["model"], "predict_proba"):
        confidence = float(artifact["model"].predict_proba(row).max() * 100)
    return crop_key, round(confidence, 2), artifact


def crop_profile(crop_key):
    df = load_dataset()
    crop_rows = df[df["label"] == crop_key]
    if crop_rows.empty:
        return df[FEATURES].mean()
    return crop_rows[FEATURES].mean()


def suitability_analysis(inputs, crop_key):
    profile = crop_profile(crop_key)
    scores = []
    missing_nutrients = []
    suggestions = []

    for nutrient, label in [("N", "Nitrogen"), ("P", "Phosphorous"), ("K", "Potassium")]:
        required = profile[nutrient]
        actual = inputs[nutrient]
        ratio = min(actual / required, required / actual) if actual and required else 0
        scores.append(max(0, min(100, ratio * 100)))
        if actual < required * 0.85:
            missing_nutrients.append(label)
            suggestions.append(f"Increase {label} through soil amendments or balanced fertilizer.")

    climate_fields = ["temperature", "humidity", "ph", "rainfall"]
    for field in climate_fields:
        required = profile[field]
        actual = inputs[field]
        tolerance = max(abs(required) * 0.25, 1)
        score = max(0, 100 - (abs(actual - required) / tolerance) * 35)
        scores.append(min(100, score))

    suitability_score = round(sum(scores) / len(scores), 2)
    status = "Suitable" if suitability_score >= 70 else "Not Suitable"
    if not suggestions:
        suggestions.append("Current nutrients are close to the crop requirement. Maintain balanced irrigation and regular monitoring.")

    soil_analysis = "NPK balance is strong." if not missing_nutrients else f"Low nutrient areas: {', '.join(missing_nutrients)}."
    climate_compatibility = "Climate conditions are compatible." if suitability_score >= 70 else "Climate or soil values need improvement before planting."

    return {
        "score": suitability_score,
        "status": status,
        "missing_nutrients": missing_nutrients,
        "soil_analysis": soil_analysis,
        "climate_compatibility": climate_compatibility,
        "suggestions": suggestions,
    }


def get_crop_id(connection, crop_key):
    row = connection.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (crop_key,)).fetchone()
    return row["crop_id"] if row else None


def save_prediction(inputs, crop_key, confidence, model_name):
    connection = get_db()
    cursor = connection.cursor()
    now = datetime.utcnow().isoformat()
    crop_id = get_crop_id(connection, crop_key)
    cursor.execute(
        """
        INSERT INTO user_predictions
        (user_id, recommended_crop_id, nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, confidence_score, model_used, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            crop_id,
            inputs["N"],
            inputs["P"],
            inputs["K"],
            inputs["temperature"],
            inputs["humidity"],
            inputs["ph"],
            inputs["rainfall"],
            confidence,
            model_name,
            now,
        ),
    )
    prediction_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO prediction_history (user_id, prediction_id, action_type, action_at) VALUES (?, ?, ?, ?)",
        (session["user_id"], prediction_id, "Crop Recommendation", now),
    )
    connection.commit()
    connection.close()
    return prediction_id


def save_suitability(inputs, crop_key, analysis, prediction_id=None):
    connection = get_db()
    cursor = connection.cursor()
    now = datetime.utcnow().isoformat()
    crop_id = get_crop_id(connection, crop_key)
    if prediction_id is None:
        cursor.execute(
            """
            INSERT INTO user_predictions
            (user_id, recommended_crop_id, nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, confidence_score, model_used, predicted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session["user_id"], crop_id, inputs["N"], inputs["P"], inputs["K"], inputs["temperature"], inputs["humidity"], inputs["ph"], inputs["rainfall"], analysis["score"], "Suitability Checker", now),
        )
        prediction_id = cursor.lastrowid
    cursor.execute(
        """
        INSERT INTO crop_suitability_analyses
        (prediction_id, crop_id, suitability_score, suitability_status, missing_nutrients, soil_analysis, climate_compatibility, improvement_suggestions, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            crop_id,
            analysis["score"],
            analysis["status"],
            ", ".join(analysis["missing_nutrients"]) or "None",
            analysis["soil_analysis"],
            analysis["climate_compatibility"],
            " ".join(analysis["suggestions"]),
            now,
        ),
    )
    cursor.execute(
        "INSERT INTO prediction_history (user_id, prediction_id, action_type, action_at) VALUES (?, ?, ?, ?)",
        (session["user_id"], prediction_id, "Suitability Analysis", now),
    )
    connection.commit()
    connection.close()


def dashboard_payload():
    df = load_dataset()
    artifact = load_artifact()
    correlation = df[FEATURES].corr().round(2)
    feature_importance = []
    model = artifact["model"]
    model_core = getattr(model, "named_steps", {}).get("model", model)
    if hasattr(model_core, "feature_importances_"):
        feature_importance = [
            {"feature": feature, "value": round(float(value), 4)}
            for feature, value in zip(FEATURES, model_core.feature_importances_)
        ]
    else:
        encoded_labels = df["label"].astype("category").cat.codes
        feature_importance = [
            {"feature": feature, "value": round(float(abs(df[feature].corr(encoded_labels))), 4)}
            for feature in FEATURES
        ]

    return {
        "crop_counts": df["label"].value_counts().to_dict(),
        "nutrient_means": df[["N", "P", "K"]].mean().round(2).to_dict(),
        "rainfall": df.groupby("label")["rainfall"].mean().round(2).to_dict(),
        "temperature": df.groupby("label")["temperature"].mean().round(2).to_dict(),
        "ph": df.groupby("label")["ph"].mean().round(2).to_dict(),
        "correlation": {
            "labels": list(correlation.columns),
            "matrix": correlation.values.tolist(),
        },
        "feature_importance": feature_importance,
        "model_results": artifact.get("model_results", {}),
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        try:
            inputs = extract_inputs(request.form)
            crop_key, confidence, artifact = predict_crop(inputs)
            analysis = suitability_analysis(inputs, crop_key)
            prediction_id = save_prediction(inputs, crop_key, confidence, artifact["best_model_name"])
            save_suitability(inputs, crop_key, analysis, prediction_id)
            crop = CROP_INFO.get(crop_key, CROP_INFO["rice"])
            result = {
                "prediction_id": prediction_id,
                "crop_key": crop_key,
                "crop": crop,
                "confidence": confidence,
                "suitability": analysis,
            }
        except Exception as exc:
            flash(f"Please check the input values. Error: {exc}", "danger")
    return render_template("recommend.html", result=result)


@app.route("/suitability", methods=["GET", "POST"])
def suitability():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        try:
            crop_key = request.form.get("crop", "")
            inputs = extract_inputs(request.form)
            analysis = suitability_analysis(inputs, crop_key)
            save_suitability(inputs, crop_key, analysis)
            result = {"crop_key": crop_key, "crop": CROP_INFO.get(crop_key), "analysis": analysis}
        except Exception as exc:
            flash(f"Please check the selected crop and input values. Error: {exc}", "danger")
    return render_template("suitability.html", crops=CROP_INFO, result=result)


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    return render_template("dashboard.html", data=dashboard_payload())


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "Feedback").strip()
        message = ""
        rating = request.form.get("rating") or None
        if not name or not email :
            flash("Name and email are required.", "danger")
            return redirect(url_for("contact"))
        connection = get_db()
        connection.execute(
            """
            INSERT INTO feedback (user_id, name, email, subject, message, rating, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session["user_id"], name, email, subject, message, rating, datetime.utcnow().isoformat()),
        )
        connection.commit()
        connection.close()
        flash("Feedback submitted successfully.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    connection = get_db()
    rows = connection.execute(
        """
        SELECT p.*, c.crop_name
        FROM user_predictions p
        LEFT JOIN crops c ON c.crop_id = p.recommended_crop_id
        ORDER BY p.predicted_at DESC
        """
    ).fetchall()
    connection.close()
    return render_template("history.html", rows=rows, crop_info=CROP_INFO)


@app.route("/export-csv")
def export_csv():
    connection = get_db()
    rows = connection.execute(
        """
        SELECT p.prediction_id, c.crop_name, p.nitrogen, p.phosphorous, p.potassium, p.temperature,
               p.humidity, p.ph, p.rainfall, p.confidence_score, p.model_used, p.predicted_at
        FROM user_predictions p
        LEFT JOIN crops c ON c.crop_id = p.recommended_crop_id
        ORDER BY p.predicted_at DESC
        """
    ).fetchall()
    connection.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["prediction_id", "crop", "nitrogen", "phosphorous", "potassium", "temperature", "humidity", "ph", "rainfall", "confidence", "model", "date"])
    for row in rows:
        writer.writerow(list(row))
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=opticrop_predictions.csv"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    inputs = extract_inputs(request.json or {})
    crop_key, confidence, artifact = predict_crop(inputs)
    return jsonify({"crop": crop_key, "confidence": confidence, "model": artifact["best_model_name"]})


@app.route("/api/crops")
def api_crops():
    return jsonify(CROP_INFO)


@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(dashboard_payload())


@app.route("/api/suitability", methods=["POST"])
def api_suitability():
    payload = request.json or {}
    crop_key = payload.get("crop")
    inputs = extract_inputs(payload)
    return jsonify(suitability_analysis(inputs, crop_key))


@app.route("/api/status")
def status():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("signup"))

        db = get_db()

        existing_user = db.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

        if existing_user:
            flash("Email already registered!", "warning")
            db.close()
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        db.execute("""
            INSERT INTO users
            (full_name,email,password_hash,role,created_at)
            VALUES (?,?,?,?,?)
        """,
        (
            full_name,
            email,
            hashed_password,
            "farmer",
            datetime.now().isoformat()
        ))

        db.commit()
        db.close()

        flash("Account Created Successfully!", "success")

        return redirect(url_for("login"))

    return render_template("signup.html")



@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        db = get_db()

        user = db.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

        db.close()

        if user and check_password_hash(user["password_hash"], password):

            session["user_id"] = user["user_id"]
            session["user_name"] = user["full_name"]

            flash("Welcome " + user["full_name"], "success")

            return redirect(url_for("dashboard"))

        flash("Invalid Email or Password", "danger")

    return render_template("login.html")



@app.route("/logout")
def logout():

    session.clear()

    flash("Logged Out Successfully!", "success")

    return redirect(url_for("home"))





init_db()


if __name__ == "__main__":
    app.run(debug=True)

