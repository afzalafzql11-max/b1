from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sqlite3, cv2, numpy as np
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
DATASET = "dataset"
DB = "database.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET, exist_ok=True)

# FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS children(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        place TEXT,
        image_path TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- EMAIL (STATIC CONFIG) ----------------
EMAIL_USER = "yourgmail@gmail.com"
EMAIL_PASS = "yourapppassword"

def send_email_alert(name, age, place, receiver):
    msg = MIMEText(f"""
MATCH FOUND!

Name: {name}
Age: {age}
Place: {place}
""")

    msg["Subject"] = "Missing Child Found"
    msg["From"] = EMAIL_USER
    msg["To"] = receiver

    try:
        server = smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(EMAIL_USER,EMAIL_PASS)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email failed:", e)

# ---------------- FACE ----------------
def extract_face(path):
    img = cv2.imread(path)
    if img is None: return None

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces)==0: return None

    x,y,w,h = faces[0]
    face = gray[y:y+h,x:x+w]
    return cv2.resize(face,(200,200))

# ---------------- AGE PROGRESSION ----------------
def age_progression(face):
    blur = cv2.GaussianBlur(face,(7,7),0)
    aged = cv2.addWeighted(face,0.6,blur,0.4,0)
    return cv2.equalizeHist(aged)

# ---------------- REVERSE AGE (YOUNGER) ----------------
def reverse_age(face):
    smooth = cv2.bilateralFilter(face,9,75,75)
    bright = cv2.convertScaleAbs(smooth, alpha=1.2, beta=10)
    return cv2.equalizeHist(bright)

# ---------------- TRAIN ----------------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8)

    faces, labels = [], []

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id,image_path FROM children")
    rows = cur.fetchall()
    conn.close()

    for r in rows:
        img = cv2.imread(r[1],0)
        if img is None: continue

        faces.append(img)
        labels.append(r[0])

        # add variations
        faces.append(age_progression(img))
        labels.append(r[0])

        faces.append(reverse_age(img))
        labels.append(r[0])

    if len(faces)==0:
        return None

    recognizer.train(faces,np.array(labels))
    return recognizer

# ---------------- SIGNUP ----------------
@app.route("/signup",methods=["POST"])
def signup():
    data = request.json

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                (data["name"],data["email"],data["password"]))
    conn.commit()
    conn.close()

    return jsonify({"message":"Account created"})

# ---------------- LOGIN ----------------
@app.route("/login",methods=["POST"])
def login():
    data = request.json

    if data["email"] == "missing child" and data["password"] == "ths345$":
        return jsonify({"status":"admin"})

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=? AND password=?",
                (data["email"],data["password"]))
    user = cur.fetchone()
    conn.close()

    if user:
        return jsonify({"status":"user","email":user[2]})
    else:
        return jsonify({"status":"fail"})

# ---------------- CROSSCHECK ----------------
@app.route("/crosscheck",methods=["POST"])
def crosscheck():
    photo = request.files["photo"]
    user_email = request.form.get("user_email")

    path = os.path.join(UPLOAD_FOLDER,photo.filename)
    photo.save(path)

    face = extract_face(path)
    if face is None:
        return jsonify({"status":"no face"})

    model = train_model()
    if model is None:
        return jsonify({"status":"no data"})

    label, conf = model.predict(face)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT name,age,place FROM children WHERE id=?",(label,))
    row = cur.fetchone()
    conn.close()

    if conf < 65:
        if user_email:
            send_email_alert(row[0],row[1],row[2],user_email)

        return jsonify({
            "status":"found",
            "confidence":float(conf),
            "name":row[0],
            "age":row[1],
            "place":row[2]
        })
    else:
        return jsonify({"status":"not found"})
