import pickle
import sqlite3
import re
from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import os
import time
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from functools import wraps
from dotenv import load_dotenv
load_dotenv()
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

# =========================
# LOAD MODELS
# =========================
model = pickle.load(open("ai_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
pest_model = load_model("plant_disease_model.h5")

classes = [
    "Chilli_Leaf_Spot",
    "Chilli_Healthy",
    "Corn_Healthy",
    "Corn_Rust",
    "Rice_Bacterial_Blight",
    "Rice_Brown_Spot",
    "Rice_Healthy",
    "Rice_Leaf_Blast",
    "Tomato_Healthy",
    "Tomato_Leaf_Blight"
]

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
API_KEY = os.getenv("API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        # PASSWORD VALIDATION
        if not re.match(
            r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$',
            password
        ):

            return render_template(
                "register.html",
                error="Password must be at least 8 characters and include uppercase, number, and special character"
            )

        # HASH PASSWORD
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        try:

            cursor.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, hashed_password)
            )

            conn.commit()

        except sqlite3.IntegrityError:

            conn.close()

            return render_template(
                "register.html",
                error="User already registered with this email"
            )

        conn.close()

        # SUCCESS REDIRECT
        return redirect("/login?success=1")

    return render_template("register.html")
@app.route("/login", methods=["GET", "POST"])
def login():

    # GET SUCCESS MESSAGE
    success = request.args.get("success")

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        user = cursor.fetchone()

        conn.close()

        # CHECK HASHED PASSWORD
        if user and check_password_hash(user[2], password):

            session["user"] = email

            return redirect("/")

        else:

            return render_template(
                "login.html",
                error="Invalid credentials"
            )

    return render_template(
        "login.html",
        success=success
    )
# =========================
# PREDICT DISEASE
# =========================
def predict_disease(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    prediction = pest_model.predict(img)

    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    disease = classes[class_index].replace("_", " ")

    return disease, confidence

# =========================
# PESTICIDE SOLUTION
# =========================
def get_solution(disease):
    data = {
        "Tomato Leaf Blight": {
            "pesticide": "Mancozeb / Chlorothalonil",
            "desc": "Spray every 7 days"
        },
        "Corn Rust": {
            "pesticide": "Propiconazole",
            "desc": "Spray every 10 days"
        },
        "Chilli Leaf Spot": {
            "pesticide": "Copper Oxychloride",
            "desc": "Remove infected leaves + spray"
        },
        "Rice Brown Spot": {
            "pesticide": "Carbendazim",
            "desc": "Spray twice"
        },
        "Rice Leaf Blast": {
            "pesticide": "Tricyclazole",
            "desc": "Early spray needed"
        },

        "Tomato Healthy": {"pesticide": "None", "desc": "No action needed"},
        "Corn Healthy": {"pesticide": "None", "desc": "No action needed"},
        "Chilli Healthy": {"pesticide": "None", "desc": "No action needed"},
        "Rice Healthy": {"pesticide": "None", "desc": "No action needed"}
    }

    return data.get(disease, {"pesticide": "Consult expert", "desc": ""})


# =========================
# ROUTES
# =========================

@app.route("/")
@login_required
def home():
    return render_template("index.html")

# ---------------- WEATHER ----------------
@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather():

    
    if request.method == "GET":
        return render_template("weather.html", bg="normal")  # default

    
    city = request.form["city"]

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    if response.get("cod") != 200:
        return render_template("weather.html", error="Invalid city or API issue", bg="normal")

    temp = response["main"]["temp"]
    humidity = response["main"]["humidity"]
    desc = response["weather"][0]["description"]

    desc_lower = desc.lower()

    # ================= BACKGROUND LOGIC =================
    bg = "normal"

    if "rain" in desc_lower:
        bg = "rainy"

    elif "clear" in desc_lower:
        bg = "sunny"

    elif "cloud" in desc_lower:  
        bg = "sunny"  

    elif temp < 15:
        bg = "winter"

    # ================= FARMING ADVICE =================
    if "rain" in desc_lower:
        advice = "🌧️ Rain expected. Avoid irrigation and pesticide spraying."

    elif temp > 35:
        advice = "🔥 High temperature. Increase irrigation and protect crops."

    elif humidity > 80:
        advice = "💧 High humidity. Risk of fungal diseases. Monitor crops."

    elif temp < 15:
        advice = "❄️ Low temperature. Protect crops from cold."

    else:
        advice = "✅ Weather is suitable for normal farming activities."

    weather_data = {
        "temp": temp,
        "humidity": humidity,
        "desc": desc
    }

    return render_template(
        "weather.html",
        weather=weather_data,
        advice=advice,
        bg=bg  
    )
# ---------------- CROP ----------------
@app.route("/recommendcrop", methods=["GET", "POST"])
@login_required
def smart_crop():

    
    if request.method == "GET":
        return render_template("recommendcrop.html")

    
    city = request.form["city"]
    soil = request.form["soil"]

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    if "main" not in response:
        return render_template("recommend.html", error="Invalid location")

    temp = response["main"]["temp"]
    humidity = response["main"]["humidity"]

    soil_encoded = encoder.transform([soil])[0]
    prediction = model.predict([[soil_encoded, temp, humidity]])[0]

    return render_template("recommendcrop.html", crop=prediction)

# ---------------- PEST DETECTION ----------------
@app.route("/detectpest", methods=["GET"])
@login_required
def detectpest_page():
    return render_template("detectpest.html")


@app.route("/pest", methods=["POST"])
@login_required
def pest():
    if "image" not in request.files:
        return render_template("detectpest.html", error="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("detectpest.html", error="No selected file")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    filename = str(int(time.time())) + "_" + filename

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    relative_path = os.path.join("static", "uploads", filename)

    disease, confidence = predict_disease(filepath)
    solution_data = get_solution(disease)

    return render_template(
        "detectpest.html",
        pest=disease,
        confidence=confidence,
        pesticide=solution_data["pesticide"],
        solution=solution_data["desc"],
        image=relative_path
    )


# ---------------- SHOW PESTICIDE FORM ----------------
@app.route("/pesticide_form", methods=["POST"])
@login_required
def pesticide_form():
    disease = request.form["disease"].strip().replace("_", " ")

    solution_data = get_solution(disease)

    return render_template(
        "detectpest.html",
        show_form=True,

        
        pest=disease,
        pesticide=solution_data["pesticide"],
        solution=solution_data["desc"]
    )

# ---------------- CALCULATE PESTICIDE ----------------
@app.route("/calculate_pesticide", methods=["POST"])
@login_required
def calculate_pesticide():
    disease = request.form["disease"].strip().replace("_", " ")
    area = float(request.form["area"])
    unit = request.form["unit"]
    age = int(request.form["age"])

    # Convert to acres
    if unit == "sq_meter":
        area /= 4047
    elif unit == "sq_yard":
        area /= 4840
    elif unit == "gunta":
        area /= 40

    dosage_map = {
        "Tomato Leaf Blight": 2.0,
        "Chilli Leaf Spot": 1.5,
        "Corn Rust": 2.5,
        "Rice Brown Spot": 1.8
    }

    base = dosage_map.get(disease, 1.5)

    if age < 30:
        factor = 0.8
    elif age < 60:
        factor = 1.0
    else:
        factor = 1.2

    total = round(area * base * factor, 2)

    solution_data = get_solution(disease)

    return render_template(
        "detectpest.html",

        
        pest=disease,
        pesticide=solution_data["pesticide"],
        solution=solution_data["desc"],
        show_form=True,

        
        result_pesticide=total,
        interval="Every 7-10 days",
        duration="2-3 weeks"
    )
# ---------------- FERTILIZER FORM ----------------
@app.route("/fertilizer_form", methods=["POST"])
@login_required
def fertilizer_form():
    crop = request.form["crop"]
    return render_template("recommendcrop.html", show_fertilizer=True, crop=crop)
# ---------------- FERTILIZER FORM ----------------
@app.route("/calculate_fertilizer", methods=["POST"])
@login_required
def calculate_fertilizer():

    crop = request.form.get("crop", "")
    area = request.form.get("area", None)
    unit = request.form.get("unit", "acre")

    
    if not crop or not area:
        return render_template(
            "recommendcrop.html",
            error="Missing crop or area. Please try again."
        )

    area = float(area)
    crop_lower = crop.lower()

    
    if unit == "sq_meter":
        area /= 4047
    elif unit == "sq_yard":
        area /= 4840
    elif unit == "gunta":
        area /= 40

    
    fertilizer_map = {
        "rice": {"N": 100, "P": 50, "K": 50},
        "tomato": {"N": 120, "P": 60, "K": 60},
        "chilli": {"N": 100, "P": 50, "K": 50},
        "corn": {"N": 120, "P": 60, "K": 40},
        "cotton": {"N": 150, "P": 60, "K": 60},
        "groundnut": {"N": 20, "P": 40, "K": 40}
    }

    base = fertilizer_map.get(crop_lower, {"N": 50, "P": 25, "K": 25})

    return render_template(
        "recommendcrop.html",
        crop=crop,
        show_fertilizer=True,
        fert_crop=crop.title(),
        nitrogen=round(base["N"] * area, 2),
        phosphorus=round(base["P"] * area, 2),
        potassium=round(base["K"] * area, 2)
    )
@app.route("/subsidy")
@login_required
def subsidy_page():
    return render_template("subsidy.html")
@app.route("/subsidy", methods=["POST"])
@login_required
def suggest_subsidy():
    age = int(request.form["age"])
    crop = request.form["crop"]
    season = request.form["season"]
    area = float(request.form["area"])
    location = request.form["location"].lower()

    # -----------------------------
    # SMART LOGIC ENGINE (IMPROVED)
    # -----------------------------

    
    if "telangana" in location:

        if area <= 5:
            scheme = "Rythu Bharosa"
            reason = "Provides ₹5000 per acre per season to support farming investment."

        elif 18 <= age <= 59:
            scheme = "Rythu Bima"
            reason = "Gives ₹5 lakh life insurance coverage for farmer family security."

        elif crop.lower() in ["tomato", "chilli", "vegetables"]:
            scheme = "Micro Irrigation Subsidy (PMKSY)"
            reason = "Up to 90% subsidy for drip irrigation to save water."

        else:
            scheme = "Farm Mechanization Subsidy"
            reason = "50–80% subsidy on modern farming equipment."

    
    else:

        if area < 2:
            scheme = "PM-KISAN"
            reason = "₹6000 yearly support for small and marginal farmers."

        elif season.lower() == "kharif":
            scheme = "PMFBY (Crop Insurance)"
            reason = "Protects crops from monsoon-related risks."

        elif crop.lower() in ["tomato", "chilli"]:
            scheme = "Micro Irrigation Subsidy (PMKSY)"
            reason = "Ideal for water-efficient crop cultivation."

        elif age < 35:
            scheme = "Farm Mechanization Scheme"
            reason = "Supports young farmers with modern equipment."

        else:
            scheme = "PM-KISAN"
            reason = "General financial support for farming needs."

    return render_template(
        "subsidy.html",
        best_scheme=scheme,
        scheme_reason=reason
    )
@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    return redirect("/login")
@app.route("/about")
@login_required
def about():
    return render_template("about.html")
# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🚀 Starting Flask App...")
    app.run(host="0.0.0.0", port=5000)