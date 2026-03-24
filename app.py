from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sqlite3, cv2, numpy as np
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
DATASET = "dataset"
VIDEO_FOLDER = "videos"

DB = "database.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

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
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS children(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        place TEXT,
        image_path TEXT
    )""")

    conn.commit()
    conn.close()

init_db()

# ---------------- EMAIL ----------------
def send_email_alert(name, age, place):
    sender = "your_email@gmail.com"
    password = "your_app_password"
    receiver = "receiver_email@gmail.com"

    msg = MIMEText(f"""
MATCH FOUND!

Name: {name}
Age: {age}
Place: {place}
""")

    msg["Subject"] = "Missing Child Found"

    try:
        server = smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(sender,password)
        server.send_message(msg)
        server.quit()
    except:
        print("Email failed")

# ---------------- FACE ----------------
def extract_face(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces)==0: return None
    x,y,w,h = faces[0]
    face = gray[y:y+h,x:x+w]
    return cv2.resize(face,(200,200))

# ---------------- AGE PROGRESSION ----------------
def age_progression(face):
    blur = cv2.GaussianBlur(face,(5,5),0)
    sharpen = cv2.addWeighted(face,1.5,blur,-0.5,0)
    return cv2.equalizeHist(sharpen)

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
            faces.append(age_progression(img))
            labels.append(r[0])

    if len(faces)==0: return None

    recognizer.train(faces,np.array(labels))
    return recognizer

# ---------------- LOGIN ----------------
@app.route("/login",methods=["POST"])
def login():
    data = request.json

    if data["email"]=="missing child" and data["password"]=="ths345$":
        return jsonify({"status":"admin"})

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=? AND password=?",
                (data["email"],data["password"]))
    user = cur.fetchone()
    conn.close()

    return jsonify({"status":"user" if user else "fail"})

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

# ---------------- REGISTER CHILD ----------------
@app.route("/register_child",methods=["POST"])
def register_child():
    name = request.form["name"]
    age = request.form["age"]
    place = request.form["place"]
    photo = request.files["photo"]

    path = os.path.join(DATASET,photo.filename)
    photo.save(path)

    img = cv2.imread(path)
    face = extract_face(img)

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

    return jsonify([
        {"id":r[0],"name":r[1],"age":r[2],"place":r[3]}
        for r in rows
    ])

# ---------------- DELETE ----------------
@app.route("/delete_child/<int:id>",methods=["DELETE"])
def delete_child(id):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM children WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return jsonify({"message":"Deleted"})

# ---------------- IMAGE MATCH ----------------
def match_face(face):
    model = train_model(False)
    aged_model = train_model(True)

    if model is None: return None

    label,conf = model.predict(face)
    label2,conf2 = aged_model.predict(face)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT name,age,place FROM children WHERE id=?",(label,))
    row = cur.fetchone()
    conn.close()

    if conf < 60:
        send_email_alert(row[0],row[1],row[2])
        return {"type":"normal","data":row}

    elif conf2 < 75:
        send_email_alert(row[0],row[1],row[2])
        return {"type":"age","data":row}

    return None

# ---------------- CROSSCHECK IMAGE ----------------
@app.route("/crosscheck",methods=["POST"])
def crosscheck():
    file = request.files["photo"]
    path = os.path.join(UPLOAD_FOLDER,file.filename)
    file.save(path)

    img = cv2.imread(path)
    face = extract_face(img)

    if face is None:
        return jsonify({"status":"no face"})

    result = match_face(face)

    if result:
        return jsonify({
            "status":"found",
            "match_type":result["type"],
            "name":result["data"][0],
            "age":result["data"][1],
            "place":result["data"][2]
        })

    return jsonify({"status":"not found"})

# ---------------- VIDEO (CCTV SIMULATION) ----------------
@app.route("/detect_video",methods=["POST"])
def detect_video():

    file = request.files["video"]
    path = os.path.join(VIDEO_FOLDER,file.filename)
    file.save(path)

    cap = cv2.VideoCapture(path)

    found_results = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # process every 10th frame
        if frame_count % 10 != 0:
            continue

        face = extract_face(frame)
        if face is None:
            continue

        result = match_face(face)

        if result:
            found_results.append({
                "name":result["data"][0],
                "age":result["data"][1],
                "place":result["data"][2],
                "type":result["type"]
            })

    cap.release()

    if len(found_results) > 0:
        return jsonify({"status":"found","results":found_results})

    return jsonify({"status":"not found"})

# ---------------- RUN ----------------
if __name__=="__main__":
    app.run(debug=True)
