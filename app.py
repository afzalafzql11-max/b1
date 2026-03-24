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

# ---------------- FACE ----------------
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

# ---------------- EMAIL ----------------
def send_email_alert(name, age, place, receiver):

    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")

    msg = MIMEText(f"""
MATCH FOUND!

Name: {name}
Age: {age}
Place: {place}
""")

    msg["Subject"] = "Missing Child Found"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        server = smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(sender,password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email failed:", e)

# ---------------- FACE EXTRACT ----------------
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
    blur = cv2.GaussianBlur(face,(5,5),0)
    sharp = cv2.addWeighted(face,1.5,blur,-0.5,0)
    return cv2.equalizeHist(sharp)

# ---------------- TRAIN ----------------
def train_model(use_aged=False):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
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

        if use_aged:
            aged = age_progression(img)
            faces.append(aged)
            labels.append(r[0])

    if len(faces)==0: return None

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

    # ADMIN LOGIN
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

# ---------------- REGISTER CHILD ----------------
@app.route("/register_child",methods=["POST"])
def register_child():

    name = request.form["name"]
    age = request.form["age"]
    place = request.form["place"]
    photo = request.files["photo"]

    path = os.path.join(DATASET,photo.filename)
    photo.save(path)

    face = extract_face(path)
    if face is None:
        return jsonify({"message":"No face detected"})

    cv2.imwrite(path,face)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO children(name,age,place,image_path) VALUES(?,?,?,?)",
                (name,age,place,path))
    conn.commit()
    conn.close()

    return jsonify({"message":"Child registered"})

# ---------------- GET CHILDREN ----------------
@app.route("/get_children")
def get_children():

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id,name,age,place FROM children")
    rows = cur.fetchall()
    conn.close()

    return jsonify([{
        "id":r[0],
        "name":r[1],
        "age":r[2],
        "place":r[3]
    } for r in rows])

# ---------------- DELETE ----------------
@app.route("/delete_child/<int:id>",methods=["DELETE"])
def delete_child(id):

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM children WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return jsonify({"message":"Deleted"})

# ---------------- IMAGE CROSSCHECK ----------------
@app.route("/crosscheck",methods=["POST"])
def crosscheck():

    photo = request.files["photo"]
    user_email = request.form.get("user_email")

    path = os.path.join(UPLOAD_FOLDER,photo.filename)
    photo.save(path)

    face = extract_face(path)
    if face is None:
        return jsonify({"status":"no face"})

    model = train_model(False)
    aged_model = train_model(True)

    if model is None:
        return jsonify({"status":"database empty"})

    label,conf = model.predict(face)
    label2,conf2 = aged_model.predict(face)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT name,age,place FROM children WHERE id=?",(label,))
    row = cur.fetchone()
    conn.close()

    if conf < 60:
        if user_email:
            send_email_alert(row[0],row[1],row[2],user_email)

        return jsonify({
            "status":"found",
            "match_type":"normal",
            "name":row[0],
            "age":row[1],
            "place":row[2]
        })

    elif conf2 < 75:
        if user_email:
            send_email_alert(row[0],row[1],row[2],user_email)

        return jsonify({
            "status":"found",
            "match_type":"age_progression",
            "name":row[0],
            "age":row[1],
            "place":row[2]
        })

    else:
        return jsonify({"status":"not found"})

# ---------------- VIDEO DETECTION ----------------
@app.route("/detect_video",methods=["POST"])
def detect_video():

    video = request.files["video"]
    path = os.path.join(UPLOAD_FOLDER,video.filename)
    video.save(path)

    cap = cv2.VideoCapture(path)
    model = train_model()

    if model is None:
        return jsonify({"status":"no data"})

    results = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # skip frames (for speed)
        if frame_count % 10 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h,x:x+w]
            face = cv2.resize(face,(200,200))

            label, conf = model.predict(face)

            if conf < 60:
                conn = sqlite3.connect(DB)
                cur = conn.cursor()
                cur.execute("SELECT name FROM children WHERE id=?",(label,))
                row = cur.fetchone()
                conn.close()

                results.append(row[0])

    cap.release()

    if len(results) > 0:
        return jsonify({"status":"found","results":results})
    else:
        return jsonify({"status":"not found"})

# ---------------- ROOT ----------------
@app.route("/")
def home():
    return "Missing Child Detection API Running"

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
