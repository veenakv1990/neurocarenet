import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import cv2
import time
import os
import json
import random
import base64
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(page_title="NeuroHealth Unified System", layout="wide")

# ------------------------
# Constants and Configuration
# ------------------------
USERS_FILE = "users.json"
UPLOAD_BASE = "user_data"
os.makedirs(UPLOAD_BASE, exist_ok=True)

# Pre-configured doctors (from first code)
AVAILABLE_DOCTORS = ["Dr. Syam Kumar", "Dr. Devi"]

DOCTOR_CREDENTIALS = {
    "syam_kumar": {
        "name": "Dr. Syam Kumar",
        "username": "syam_kumar",
        "password": "syam123",
        "email": "syam.kumar@neurohealth.com"
    },
    "devi": {
        "name": "Dr. Devi",
        "username": "devi",
        "password": "devi123",
        "email": "devi@neurohealth.com"
    }
}

# Available hospitals and their doctors for referral (from second code)
REFERRAL_HOSPITALS = {
    "H1 - Apollo Hospital Chennai": {
        "doctors": ["Dr. Ravi Kumar - Neurologist", "Dr. Priya Sharma - Neurosurgeon"],
        "contact": "+91-44-2829-3333",
        "speciality": "Comprehensive Neurological Care"
    },
    "H2 - BLK-Max Super Speciality Hospital Delhi": {
        "doctors": ["Dr. Ashish Suri - Neurosurgeon", "Dr. Manjari Tripathi - Neurologist"],
        "contact": "+91-11-3040-3040",
        "speciality": "Advanced Neurosciences"
    },
    "H3 - Kokilaben Dhirubhai Ambani Hospital Mumbai": {
        "doctors": ["Dr. Sudhir Shah - Movement Disorders", "Dr. Nita Nair - Cognitive Neurology"],
        "contact": "+91-22-4269-6969",
        "speciality": "Specialized Brain & Spine Care"
    }
}

# Blood group options
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# MODIFIED: Feature descriptions for video analysis (60 seconds - 6 tasks, 10 seconds each)
FEATURE_GUIDELINES = [
    ("Facial expressions & blink rate", "Keep your face visible. Blink normally and show a few natural expressions."),
    ("Resting tremor & postural tremor", "Keep your hands relaxed on your lap. Observe for tremors while resting."),
    ("Finger tapping (bradykinesia)", "Tap your index finger and thumb together repeatedly for 5 seconds."),
    ("Arm swing & gait", "Stand and walk a few steps, letting your arms swing naturally."),
    ("Postural stability", "Stand still, then turn around carefully. Try not to lose balance."),
    ("Facial symmetry & head tremor", "Smile naturally, relax your face, and keep head still."),
]

# Audio feature descriptions for display
AUDIO_FEATURES = [
    ("Speech rate & fluency", "Analysis of speaking speed and smoothness of speech"),
    ("Voice quality & stability", "Assessment of voice tremor and pitch variations"),
    ("Articulation precision", "Clarity of consonants and vowel pronunciation"),
    ("Pause patterns", "Frequency and duration of speech pauses"),
    ("Monotonicity", "Variation in pitch and tone during speech"),
    ("Word finding ability", "Ease of retrieving and expressing words"),
    ("Semantic coherence", "Logical flow and meaning in speech"),
    ("Memory recall", "Ability to remember and repeat information"),
    ("Cognitive processing", "Speed of verbal responses and comprehension"),
    ("Neurological speech markers", "Signs of motor speech disorders"),
]

# MODIFIED: Audio recorder HTML component (60 seconds max)
AUDIO_RECORDER_HTML = r"""
<div style="font-family: sans-serif; margin:6px;">
  <div style="display:flex; gap:12px; align-items:center;">
    <button id="recBtn">🎙 Start Recording</button>
    <button id="stopBtn" style="display:none; background:#dc3545; color:white;">⏹ Stop Recording</button>
    <div id="timer" style="font-weight:bold; font-size:16px;">00:00</div>
    <div id="status" style="margin-left:8px;color:#555;font-size:13px;">Idle</div>
  </div>
  <canvas id="wavecanvas" width="600" height="100" style="width:100%; border-radius:6px; background:#fafafa; margin-top:8px; border:1px solid #ddd;"></canvas>
  <div style="margin-top:6px; font-size:12px; color:#666;">Make sure to allow microphone access when prompted.</div>
  <div id="taskDisplay" style="margin-top:6px; font-size:14px; font-weight:bold; color:#1f618d;">Task: Waiting to start...</div>
</div>

<script>
function nowStr() { return Date.now(); }
function sendToStreamlit(obj) {
  try { if (typeof Streamlit !== "undefined" && Streamlit.setComponentValue) { Streamlit.setComponentValue(obj); return; } } catch(e){}
  try { window.parent.postMessage(obj, "*"); } catch(e){}
}

let recBtn = document.getElementById("recBtn");
let stopBtn = document.getElementById("stopBtn");
let timerEl = document.getElementById("timer");
let statusEl = document.getElementById("status");
let canvas = document.getElementById("wavecanvas");
let taskDisplay = document.getElementById("taskDisplay");
let ctx = canvas.getContext("2d");

let mediaRecorder = null;
let gumStream = null;
let audioCtx = null;
let analyser = null;
let dataArray = null;
let bufferLength = 0;
let drawId = null;
let elapsed = 0;
let timerInterval = null;
let chunks = [];
let maxDuration = 60; // MODIFIED: 60 seconds max

// MODIFIED: 4 tasks, 15 seconds each
const tasks = [
  "Read aloud: 'The quick brown fox jumps over the lazy dog'",
  "Count clearly: 'One, two, three ... up to fifteen'",
  "Say three fruits you like",
  "Describe what you see around you"
];

function formatTime(sec) {
  let m = Math.floor(sec/60);
  let s = sec % 60;
  return String(m).padStart(2,'0') + ":" + String(s).padStart(2,'0');
}

function updateTask(elapsed){
  if(elapsed < 15) taskDisplay.innerText = "Task: " + tasks[0];
  else if(elapsed < 30) taskDisplay.innerText = "Task: " + tasks[1];
  else if(elapsed < 45) taskDisplay.innerText = "Task: " + tasks[2];
  else taskDisplay.innerText = "Task: " + tasks[3];
}

function drawWave() {
  drawId = requestAnimationFrame(drawWave);
  if (!analyser) return;
  analyser.getByteTimeDomainData(dataArray);
  ctx.fillStyle = "#fafafa";
  ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#2c3e50";
  ctx.beginPath();
  let sliceWidth = canvas.width / bufferLength;
  let x = 0;
  for (let i = 0; i < bufferLength; i++) {
    let v = dataArray[i] / 128.0;
    let y = v * canvas.height/2;
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    x += sliceWidth;
  }
  ctx.lineTo(canvas.width, canvas.height/2);
  ctx.stroke();
}

async function startRecording() {
  try {
    recBtn.disabled = true;
    recBtn.style.display = "none";
    stopBtn.style.display = "inline-block";
    statusEl.innerText = "Requesting microphone...";
    elapsed = 0;
    timerEl.innerText = formatTime(0);
    chunks = [];

    gumStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    statusEl.innerText = "Recording... Click Stop to finish";

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(gumStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    bufferLength = analyser.fftSize;
    dataArray = new Uint8Array(bufferLength);
    source.connect(analyser);

    drawWave();

    mediaRecorder = new MediaRecorder(gumStream);
    mediaRecorder.ondataavailable = function(e) { if (e.data && e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = function() {
      cancelAnimationFrame(drawId);
      if (audioCtx && audioCtx.state !== "closed") { try { audioCtx.close(); } catch(e){} }
      statusEl.innerText = "Recording complete!";
      clearInterval(timerInterval);
     
      // Reset buttons
      recBtn.disabled = false;
      recBtn.style.display = "inline-block";
      stopBtn.style.display = "none";

      const blob = new Blob(chunks, { type: chunks[0] ? chunks[0].type : "audio/webm" });
      const mime = blob.type || "audio/webm";
      const reader = new FileReader();
      reader.onloadend = function() {
        const dataUrl = reader.result;
        const base64 = dataUrl.split(",")[1];
        const filename = "recording_" + Date.now() + "." + (mime.includes("/") ? mime.split("/")[1] : "webm");
        const payload = {
          isStreamlitMessage: true,
          type: "AUDIO_DATA",
          data: base64,
          mime: mime,
          filename: filename
        };
        sendToStreamlit(payload);
        statusEl.innerText = "Processing complete!";
      };
      reader.readAsDataURL(blob);

      try { gumStream.getTracks().forEach(t => t.stop()); } catch(e){}
    };

    mediaRecorder.start();
    timerInterval = setInterval(function(){
      elapsed += 1;
      timerEl.innerText = formatTime(elapsed);
      updateTask(elapsed);
     
      // Auto-stop after max duration
      if (elapsed >= maxDuration) {
        stopRecording();
      }
    }, 1000);
   
  } catch (err) {
    statusEl.innerText = "Error: " + err.message;
    recBtn.disabled = false;
    recBtn.style.display = "inline-block";
    stopBtn.style.display = "none";
    try { clearInterval(timerInterval); } catch(e){}
  }
}

function stopRecording() {
  try {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  } catch(e){}
}

recBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
</script>
"""

# ------------------------
# Session State Initialization
# ------------------------
def initialize_session_state():
    defaults = {
        "page": "patient_register",
        "doctor": None,
        "user": None,
        "current_visit_index": -1,
        "video_file": None,
        "video_scores": None,
        "video_probs": None,
        "audio_file": None,
        "audio_scores": None,
        "audio_probs": None,
        "start_time": None,
        "recording_active": False,
        "streamlit_message": None,
        "audio_processed": False,
        "audio_bytes": None,
        "doctor_tmp": {},
        "temp_doctor": None,
        "assessment_section": 0,  # NEW: Track current assessment section
    }
   
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# ------------------------
# Helper Functions
# ------------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            try:
                users = json.load(f)
                for email, user in users.items():
                    if "patient_id" not in user:
                        user["patient_id"] = generate_unique_patient_id(users)
                save_users(users)
                return users
            except Exception:
                return {}
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def generate_unique_patient_id(users):
    existing_ids = {user.get("patient_id") for user in users.values() if user.get("patient_id")}
    while True:
        patient_id = f"{random.randint(100000, 999999):06d}"
        if patient_id not in existing_ids:
            return patient_id

def validate_phone_number(phone):
    """Validate phone number - must be exactly 10 digits"""
    if not phone:
        return False, "Phone number is required"
   
    # Remove any spaces or special characters
    phone_clean = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("+", "")
   
    # Check if it contains only digits
    if not phone_clean.isdigit():
        return False, "Phone number must contain only numbers"
   
    # Check if it's exactly 10 digits
    if len(phone_clean) != 10:
        return False, "Phone number must be exactly 10 digits"
   
    return True, phone_clean

def safe_selectbox_index(options_list, value, default):
    """Safely find the index of a value in a list, falling back to a default."""
    try:
        if value in options_list:
            return options_list.index(value)
        else:
            return options_list.index(default) if default in options_list else 0
    except ValueError:
        return 0

def save_audio_file(file_bytes, filename, user_id="guest"):
    user_dir = os.path.join(UPLOAD_BASE, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, filename)
    with open(path, "wb") as fh:
        fh.write(file_bytes)
    return path

# Analysis functions (mock implementations)
def analyze_frame(frame):
    return {
        "Facial expressions & blink rate": np.random.rand(),
        "Resting tremor & postural tremor": np.random.rand(),
        "Finger tapping (bradykinesia)": np.random.rand(),
        "Arm swing & gait": np.random.rand(),
        "Postural stability": np.random.rand(),
        "Facial symmetry & head tremor": np.random.rand(),
    }

def analyze_audio_simple():
    return {
        "Speech rate & fluency": np.random.rand(),
        "Voice quality & stability": np.random.rand(),
        "Articulation precision": np.random.rand(),
        "Pause patterns": np.random.rand(),
        "Monotonicity": np.random.rand(),
        "Word finding ability": np.random.rand(),
        "Semantic coherence": np.random.rand(),
        "Memory recall": np.random.rand(),
        "Cognitive processing": np.random.rand(),
        "Neurological speech markers": np.random.rand(),
    }

def compute_video_probabilities(avg_scores):
    risk_score = np.mean(list(avg_scores.values()))
    
    # Adjust for more realistic screening - most people should be normal
    # Use inverse relationship: lower risk_score = higher normal probability
    normal_prob = max(0.3, 1 - risk_score * 1.2)  # Minimum 30% normal chance
    
    # Distribute remaining probability among conditions
    remaining_prob = max(0.1, 1 - normal_prob)
    
    probs = {
        "Normal": round(normal_prob, 2),
        "Parkinson's": round(remaining_prob * 0.35, 2),
        "Stroke": round(remaining_prob * 0.25, 2),
        "Alzheimer's": round(remaining_prob * 0.25, 2),
        "Brain Tumor": round(remaining_prob * 0.15, 2)
    }
    
    # Normalize to ensure sum = 1
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v/total, 2) for k, v in probs.items()}
    
    return probs

def compute_audio_probabilities(avg_scores):
    risk_score = np.mean(list(avg_scores.values()))
    
    # Similar adjustment for audio - favor normal cases
    normal_prob = max(0.35, 1 - risk_score * 1.1)  # Minimum 35% normal chance
    
    # Distribute remaining probability among conditions
    remaining_prob = max(0.1, 1 - normal_prob)
    
    probs = {
        "Normal": round(normal_prob, 2),
        "Parkinson's": round(remaining_prob * 0.30, 2),
        "Alzheimer's": round(remaining_prob * 0.30, 2),
        "Stroke": round(remaining_prob * 0.25, 2),
        "Brain Tumor": round(remaining_prob * 0.15, 2)
    }
    
    # Normalize to ensure sum = 1
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v/total, 2) for k, v in probs.items()}
    
    return probs

def combine_predictions(video_probs, audio_probs):
    combined = {}
    video_weight = 0.6
    audio_weight = 0.4
   
    all_conditions = set(list(video_probs.keys()) + list(audio_probs.keys()))
   
    for condition in all_conditions:
        video_score = video_probs.get(condition, 0)
        audio_score = audio_probs.get(condition, 0)
        combined[condition] = round(video_score * video_weight + audio_score * audio_weight, 2)
   
    total = sum(combined.values())
    if total > 0:
        combined = {k: round(v/total, 2) for k, v in combined.items()}
   
    return combined

def create_radar_chart(scores, title="Feature Scores"):
    categories = list(scores.keys())
    values = list(scores.values())
   
    categories += [categories[0]]
    values += [values[0]]
   
    fig = px.line_polar(
        r=values,
        theta=categories,
        line_close=True,
        title=title,
        range_r=[0, 1]
    )
    fig.update_traces(fill='toself')
    return fig

# ------------------------
# PAGE IMPLEMENTATIONS
# ------------------------

def page_patient_register():
    st.title("NeuroHealth Primary Health Care Center")
    st.header("Patient Registration")
   
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Welcome! Please register as a new patient.")
       
        name = st.text_input("Patient Full Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=0)
        blood_group = st.selectbox("Blood Group", ["Select Blood Group"] + BLOOD_GROUPS)
        phone = st.text_input("Phone Number")
       
        st.markdown("### Select Your Doctor")
        selected_doctor = st.selectbox("Choose Doctor", ["Select Doctor"] + AVAILABLE_DOCTORS)
       
        if selected_doctor != "Select Doctor":
            st.success(f"You will be assigned to: {selected_doctor}")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register Patient", use_container_width=True):
                # Validate phone number
                phone_valid, phone_message = validate_phone_number(phone)
               
                if not all([name, blood_group != "Select Blood Group", selected_doctor != "Select Doctor"]):
                    st.error("Please fill all required fields and select a doctor")
                elif not phone_valid:
                    st.error(f"Invalid phone number: {phone_message}")
                else:
                    users = load_users()
                    patient_id = generate_unique_patient_id(users)
                   
                    patient_email = f"patient_{patient_id}@neurohealth.com"
                   
                    users[patient_email] = {
                        "patient_id": patient_id,
                        "name": name,
                        "age": int(age),
                        "blood_group": blood_group,
                        "phone": phone_message,  # Use cleaned phone number
                        "assigned_doctor": selected_doctor,
                        "visits": [],
                        "created_date": str(date.today())
                    }
                    save_users(users)
                   
                    st.session_state["user"] = users[patient_email]
                    st.success(f"Patient registered successfully! Patient ID: {patient_id}")
                    st.success(f"Assigned Doctor: {selected_doctor}")
                    time.sleep(2)
                    st.session_state["page"] = "home"
                    st.rerun()
       
        with col_b:
            if st.button("Doctor Login", use_container_width=True):
                st.session_state["page"] = "doctor_login"
                st.rerun()

def page_doctor_login():
    st.title("Doctor Login")
    st.header("Access Doctor Portal")
   
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Login to view your patient records")
       
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Login", use_container_width=True):
                doctor_found = None
                for doc_key, doc_info in DOCTOR_CREDENTIALS.items():
                    if doc_info["username"] == username and doc_info["password"] == password:
                        doctor_found = doc_info
                        break
               
                if doctor_found:
                    st.session_state["doctor"] = doctor_found
                    st.session_state["page"] = "doctor_dashboard"
                    st.success(f"Login successful! Welcome {doctor_found['name']}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please check username and password.")
       
        with col_b:
            if st.button("Back to Patient Registration", use_container_width=True):
                st.session_state["page"] = "patient_register"
                st.rerun()

def page_doctor_dashboard():
    st.title("Doctor Dashboard")
   
    if "doctor" in st.session_state and st.session_state["doctor"]:
        doctor = st.session_state["doctor"]
        st.header(f"Welcome, {doctor['name']}")
    else:
        st.error("Doctor not logged in. Please login first.")
        st.session_state["page"] = "doctor_login"
        st.rerun()
        return
   
    # Load patients assigned to this doctor
    users = load_users()
    doctor_patients = []
   
    for user_email, user_data in users.items():
        if user_data.get("assigned_doctor") == doctor["name"]:
            doctor_patients.append((user_email, user_data))
   
    # Display statistics
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.metric("Total Patients", len(doctor_patients))
   
    with col2:
        patients_with_visits = sum(1 for _, patient in doctor_patients if patient.get("visits"))
        st.metric("Patients with Visits", patients_with_visits)
   
    with col3:
        total_visits = sum(len(patient.get("visits", [])) for _, patient in doctor_patients)
        st.metric("Total Visits", total_visits)
   
    st.markdown("---")
    st.subheader(f"Your Patients ({len(doctor_patients)} total)")
   
    if doctor_patients:
        for user_email, patient in doctor_patients:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
               
                with col1:
                    st.write(f"**{patient.get('name')}**")
                    st.write(f"ID: {patient.get('patient_id')}")
               
                with col2:
                    st.write(f"Age: {patient.get('age')} years")
                    st.write(f"Blood: {patient.get('blood_group')}")
               
                with col3:
                    st.write(f"Phone: {patient.get('phone')}")
                    visits_count = len(patient.get('visits', []))
                    st.write(f"Visits: {visits_count}")
               
                with col4:
                    if st.button("Select", key=f"select_{patient.get('patient_id')}"):
                        st.session_state["user"] = patient
                        st.session_state["page"] = "home"
                        st.rerun()
               
                st.markdown("---")
    else:
        st.info("No patients assigned to you yet.")
   
    # Quick actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register New Patient", use_container_width=True):
            st.session_state["page"] = "patient_register_by_doctor"
            st.rerun()
   
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state["doctor"] = None
            st.session_state["page"] = "patient_register"
            st.rerun()

def page_patient_register_by_doctor():
    st.title("Register New Patient")
   
    if "doctor" in st.session_state and st.session_state["doctor"]:
        doctor = st.session_state["doctor"]
        st.header(f"Registering under {doctor['name']}")
    else:
        st.error("Doctor not logged in.")
        return
   
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name = st.text_input("Patient Full Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        blood_group = st.selectbox("Blood Group", ["Select Blood Group"] + BLOOD_GROUPS)
        phone = st.text_input("Phone Number")
       
        st.info(f"Will be assigned to: {doctor['name']}")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register Patient", use_container_width=True):
                # Validate phone number
                phone_valid, phone_message = validate_phone_number(phone)
               
                if not all([name, blood_group != "Select Blood Group"]):
                    st.error("Please fill all required fields")
                elif not phone_valid:
                    st.error(f"Invalid phone number: {phone_message}")
                else:
                    users = load_users()
                    patient_id = generate_unique_patient_id(users)
                   
                    patient_email = f"patient_{patient_id}@neurohealth.com"
                   
                    users[patient_email] = {
                        "patient_id": patient_id,
                        "name": name,
                        "age": int(age),
                        "blood_group": blood_group,
                        "phone": phone_message,  # Use cleaned phone number
                        "assigned_doctor": doctor['name'],
                        "visits": [],
                        "created_date": str(date.today())
                    }
                    save_users(users)
                   
                    st.session_state["user"] = users[patient_email]
                    st.success(f"Patient registered successfully! Patient ID: {patient_id}")
                    time.sleep(1)
                    st.session_state["page"] = "home"
                    st.rerun()
       
        with col_b:
            if st.button("Back to Dashboard", use_container_width=True):
                st.session_state["page"] = "doctor_dashboard"
                st.rerun()

def page_home():
    st.title("NeuroHealth Dashboard")
   
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        st.header(f"Welcome, {user['name']}!")
       
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(f"Patient ID: {user.get('patient_id', 'N/A')}")
            st.info(f"Assigned Doctor: {user.get('assigned_doctor', 'N/A')}")
           
            if st.button("Patient Information", use_container_width=True):
                st.session_state["page"] = "patient_info"
                st.rerun()
               
            if st.button("View Medical Visits", use_container_width=True):
                st.session_state["page"] = "visiting_data"
                st.rerun()
           
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Change Patient", use_container_width=True):
                    st.session_state["user"] = None
                    if st.session_state.get("doctor"):
                        st.session_state["page"] = "doctor_dashboard"
                    else:
                        st.session_state["page"] = "patient_register"
                    st.rerun()
           
            with col_b:
                if st.button("Doctor Login", use_container_width=True):
                    st.session_state["page"] = "doctor_login"
                    st.rerun()

def page_patient_info():
    st.title("Patient Information")
   
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        st.header(f"Profile: {user['name']} (ID: {user.get('patient_id', 'N/A')})")
       
        col1, col2 = st.columns([2, 1])
        with col1:
            name = st.text_input("Full Name", value=user.get("name", ""))
            age = st.number_input("Age", value=user.get("age", 0), min_value=0, step=1)
           
            blood_groups = ["Select Blood Group"] + BLOOD_GROUPS
            current_bg = user.get("blood_group", "Select Blood Group")
            try:
                bg_index = blood_groups.index(current_bg)
            except ValueError:
                bg_index = 0
            blood_group = st.selectbox("Blood Group", blood_groups, index=bg_index)
           
            phone = st.text_input("Phone Number", value=user.get("phone", ""))
           
            st.info(f"Assigned Doctor: {user.get('assigned_doctor', 'N/A')}")

            if st.button("Save Changes"):
                # Validate phone number
                phone_valid, phone_message = validate_phone_number(phone)
               
                if not phone_valid:
                    st.error(f"Invalid phone number: {phone_message}")
                else:
                    users = load_users()
                    user_key = None
                    for key, u in users.items():
                        if u.get("patient_id") == user.get("patient_id"):
                            user_key = key
                            break
                   
                    if user_key:
                        users[user_key].update({
                            "name": name,
                            "age": int(age),
                            "blood_group": blood_group if blood_group != "Select Blood Group" else user.get("blood_group"),
                            "phone": phone_message  # Use cleaned phone number
                        })
                        save_users(users)
                        st.session_state["user"] = users[user_key]
                        st.success("Profile updated successfully!")

        with col2:
            st.info("Quick Actions")
           
            if st.button("Add New Visit", use_container_width=True):
                users = load_users()
                user_key = None
                for key, u in users.items():
                    if u.get("patient_id") == user.get("patient_id"):
                        user_key = key
                        break
               
                if user_key:
                    visit = {
                        "date": str(date.today()),
                        "reason": "",
                        "hospital": "Primary Health Care Center",
                        "doctor": user.get("assigned_doctor")
                    }
                    users[user_key]["visits"].append(visit)
                    save_users(users)
                    st.session_state["user"] = users[user_key]
                    st.session_state["current_visit_index"] = len(users[user_key]["visits"]) - 1
                    st.session_state["page"] = "select_facility"
                    st.rerun()
           
            if st.button("Back to Home", use_container_width=True):
                st.session_state["page"] = "home"
                st.rerun()

def page_visiting_data():
    st.title("Medical Visit History")
   
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        st.header(f"Visits for {user['name']} (ID: {user.get('patient_id', 'N/A')})")
       
        if "visits" in user and user["visits"]:
            for idx, visit in enumerate(user["visits"]):
                with st.container():
                    st.markdown(f"### Visit #{idx+1}")
                    col1, col2, col3 = st.columns([2, 1, 1])
                   
                    with col1:
                        st.write(f"**Date:** {visit.get('date')}")
                        st.write(f"**Reason:** {visit.get('reason', 'Not specified')}")
                        st.write(f"**Hospital:** {visit.get('hospital', 'Primary Health Care Center')}")
                        st.write(f"**Doctor:** {visit.get('doctor', user.get('assigned_doctor'))}")
                   
                    with col2:
                        if st.button(f"Edit Visit {idx+1}", key=f"edit_{idx}"):
                            st.session_state["current_visit_index"] = idx
                            st.session_state["assessment_section"] = 0  # Reset to first section
                            st.session_state["page"] = "select_facility"
                            st.rerun()
                   
                    with col3:
                        if st.button(f"Delete {idx+1}", key=f"del_{idx}"):
                            users = load_users()
                            user_key = None
                            for key, u in users.items():
                                if u.get("patient_id") == user.get("patient_id"):
                                    user_key = key
                                    break
                           
                            if user_key:
                                users[user_key]["visits"].pop(idx)
                                save_users(users)
                                st.session_state["user"] = users[user_key]
                                st.rerun()
                   
                    st.markdown("---")
        else:
            st.info("No visits recorded yet. Add a new visit to get started.")

        if st.button("Back to Home"):
            st.session_state["page"] = "home"
            st.rerun()

def page_select_facility():
    st.title("Visit Information")
   
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        visit_index = st.session_state.get("current_visit_index", -1)
       
        if visit_index < 0 or visit_index >= len(user.get("visits", [])):
            st.error("No visit selected.")
            return
           
        visit = user["visits"][visit_index]
       
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Visit Information")
            reason = st.text_input("Reason for Visit", value=visit.get("reason", ""))
           
            st.info("Hospital: Primary Health Care Center")
            st.info(f"Consulting Doctor: {user.get('assigned_doctor')}")
           
            if st.button("Save & Continue to Assessment"):
                users = load_users()
                user_key = None
                for key, u in users.items():
                    if u.get("patient_id") == user.get("patient_id"):
                        user_key = key
                        break
               
                if user_key:
                    users[user_key]["visits"][visit_index].update({
                        "reason": reason,
                        "hospital": "Primary Health Care Center",
                        "doctor": user.get('assigned_doctor'),
                        "status": "completed"
                    })
                    save_users(users)
                    st.session_state["user"] = users[user_key]
                    st.session_state["assessment_section"] = 0  # Start with first section
                    st.success("Visit information saved successfully!")
                    time.sleep(1)
                    st.session_state["page"] = "doctor_assessment"
                    st.rerun()
       
        with col2:
            st.info("Visit Details")
            st.write(f"**Patient:** {user['name']}")
            st.write(f"**Patient ID:** {user.get('patient_id', 'N/A')}")
            st.write(f"**Date:** {visit.get('date')}")
            st.write(f"**Assigned Doctor:** {user.get('assigned_doctor')}")
           
            if st.button("Back to Visits"):
                st.session_state["page"] = "visiting_data"
                st.rerun()

def page_doctor_assessment():
    st.title("Doctor's Clinical Assessment")
   
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        visit_index = st.session_state.get("current_visit_index", -1)
       
        if visit_index < 0 or visit_index >= len(user.get("visits", [])):
            st.error("No visit selected.")
            return
           
        visit = user["visits"][visit_index]
        st.write(f"Assessment for Visit on {visit.get('date')} — Doctor: {user.get('assigned_doctor')}")

        # Get current section (default to 0)
        current_section = st.session_state.get("assessment_section", 0)
       
        # Define section titles and total sections
        section_titles = [
            "General Information",
            "Medical History",
            "Clinical Observations",
            "Test Results",
            "MRI/Imaging & Final"
        ]
        total_sections = len(section_titles)
       
        # Progress indicator
        st.progress((current_section + 1) / total_sections)
        st.markdown(f"### Section {current_section + 1} of {total_sections}: {section_titles[current_section]}")
       
        # Initialize doctor_tmp if not present
        if "doctor_tmp" not in st.session_state:
            st.session_state["doctor_tmp"] = visit.get("doctor_assessment", {})

        # Section-specific content
        if current_section == 0:  # General Info
            st.subheader("Patient Information")
            col1, col2 = st.columns(2)
           
            with col1:
                st.write(f"**Patient ID:** {user.get('patient_id', 'N/A')}")
                name = st.text_input("Patient Name", value=user.get("name",""))
                age = st.number_input("Age", min_value=0, value=user.get("age",0))
               
                genders = ["Male","Female","Other"]
                gender = st.selectbox("Gender", genders,
                                      index=safe_selectbox_index(genders, user.get("gender"), "Male"))
               
                st.write(f"**Blood Group:** {user.get('blood_group', 'N/A')}")
           
            with col2:
                education = st.selectbox("Education Level", ["No formal","School","Graduate","Postgraduate"])
                fam_hist = st.multiselect("Family Medical History", ["None","Alzheimer's","Parkinson's","Stroke","Brain Tumor"])
                lifestyle = st.multiselect("Lifestyle Factors", ["Smoking","Alcohol","Physical activity","Sleep issues"])
           
            st.session_state["doctor_tmp"].update({
                "patient_id": user.get("patient_id", "N/A"), "name": name, "age": int(age), "gender": gender,
                "blood_group": user.get("blood_group", "N/A"), "education": education,
                "family_history": fam_hist, "lifestyle": lifestyle
            })

        elif current_section == 1:  # Medical History
            st.subheader("Medical History")
            col1, col2 = st.columns(2)
           
            with col1:
                htn = st.radio("Hypertension", ["Yes","No"], index=1)
                diabetes = st.radio("Diabetes", ["Yes","No"], index=1)
                hyperlipidemia = st.radio("Hyperlipidemia", ["Yes","No"], index=1)
                atrial_fibrillation = st.radio("Atrial Fibrillation", ["Yes","No"], index=1)
                prior_tia_stroke = st.radio("Prior TIA/Stroke", ["Yes","No"], index=1)
                acute_onset = st.radio("Acute Onset History", ["Yes","No"], index=1)
           
            with col2:
                carotid_bruit = st.radio("Carotid Bruit", ["Yes","No"], index=1)
                cardio = st.radio("Cardiovascular Disease", ["Yes","No"], index=1)
                head_injury = st.radio("Previous Head Injury", ["Yes","No"], index=1)
                seizures = st.radio("Seizures/Fainting", ["Yes","No"], index=1)
                psych = st.text_input("Psychiatric History")
                meds = st.text_area("Current Medications")
           
            st.session_state["doctor_tmp"].update({
                "htn": htn, "diabetes": diabetes, "hyperlipidemia": hyperlipidemia,
                "atrial_fibrillation": atrial_fibrillation, "prior_tia_stroke": prior_tia_stroke,
                "acute_onset": acute_onset, "carotid_bruit": carotid_bruit,
                "cardio": cardio, "psych_history": psych, "medications": meds,
                "head_injury": head_injury, "seizures": seizures
            })

        elif current_section == 2:  # Clinical Observations
            st.subheader("Clinical Observations")
            col1, col2 = st.columns(2)
           
            with col1:
                memory = st.selectbox("Memory Loss", ["None","Mild","Moderate","Severe"])
                orientation_deficit = st.selectbox("Orientation Deficit", ["None","Mild","Moderate","Severe"])
                speech_issues = st.selectbox("Speech Issues", ["Normal","Slurred","Slow","Word-finding difficulty"])
                aphasia = st.selectbox("Aphasia", ["None","Mild","Moderate","Severe"])
                dysarthria = st.selectbox("Dysarthria", ["None","Mild","Moderate","Severe"])
                tremors = st.selectbox("Tremors", ["None","Mild","Moderate","Severe"])
                rigidity = st.selectbox("Rigidity", ["None","Mild","Moderate","Severe"])
                hemiparesis = st.selectbox("Hemiparesis", ["None","Mild","Moderate","Severe"])
                gait = st.selectbox("Gait Problems", ["Normal","Slow","Unsteady","Falls"])
                facial = st.selectbox("Facial Expression", ["Normal","Reduced","Masked"])
           
            with col2:
                headaches = st.selectbox("Headaches", ["None","Mild","Moderate","Severe"])
                vision_hearing = st.text_input("Vision/Hearing Issues")
                handwriting = st.selectbox("Handwriting (Micrographia)", ["None","Mild","Moderate","Severe"])
                set_shifting = st.selectbox("Set-Shifting Difficulty", ["None","Mild","Moderate","Severe"])
                planning_impairment = st.selectbox("Planning Impairment", ["None","Mild","Moderate","Severe"])
                iadl_decline = st.selectbox("IADL Decline", ["None","Mild","Moderate","Severe"])
                adl_decline = st.selectbox("ADL Decline", ["None","Mild","Moderate","Severe"])
                apathy_scale = st.slider("Apathy Scale (0-10)", 0, 10, 0)
                depression_scale = st.slider("Depression Scale (0-10)", 0, 10, 0)
                motor_weakness = st.text_input("Motor Weakness (describe)")
           
            st.session_state["doctor_tmp"].update({
                "memory": memory, "orientation_deficit": orientation_deficit,
                "speech_issues": speech_issues, "aphasia": aphasia, "dysarthria": dysarthria,
                "tremors": tremors, "rigidity": rigidity, "hemiparesis": hemiparesis,
                "gait": gait, "facial": facial, "motor_weakness": motor_weakness,
                "headaches": headaches, "vision_hearing": vision_hearing,
                "handwriting": handwriting, "set_shifting": set_shifting,
                "planning_impairment": planning_impairment, "iadl_decline": iadl_decline,
                "adl_decline": adl_decline, "apathy_scale": apathy_scale,
                "depression_scale": depression_scale
            })

        elif current_section == 3:  # Test Results
            st.subheader("Cognitive & Motor Test Scores")
            col1, col2 = st.columns(2)
           
            with col1:
                mmse = st.number_input("MMSE Score (0–30)", min_value=0, max_value=30, value=0)
                moca = st.number_input("MoCA Score (0–30)", min_value=0, max_value=30, value=0)
                delayed_recall = st.selectbox("Delayed Recall Impairment", ["None","Mild","Moderate","Severe"])
                recognition_memory = st.number_input("Recognition Memory Errors", min_value=0, max_value=10, value=0)
                clock_drawing = st.number_input("Clock Drawing Errors", min_value=0, max_value=5, value=0)
           
            with col2:
                trails_b_time = st.number_input("Trails B Time (seconds)", min_value=0, value=0)
                stroop_errors = st.number_input("Stroop Errors", min_value=0, max_value=10, value=0)
                updrs = st.number_input("UPDRS Score", min_value=0, value=0)
                nihss = st.number_input("NIH Stroke Scale", min_value=0, value=0)
           
            st.session_state["doctor_tmp"].update({
                "mmse": int(mmse), "moca": int(moca), "delayed_recall": delayed_recall,
                "recognition_memory": int(recognition_memory), "clock_drawing": int(clock_drawing),
                "trails_b_time": int(trails_b_time), "stroop_errors": int(stroop_errors),
                "updrs": int(updrs), "nihss": int(nihss)
            })

        elif current_section == 4:  # MRI/Imaging & Final
            st.subheader("Imaging Findings")
            white_matter_lesions = st.selectbox("White-Matter Lesions", ["None","Mild","Moderate","Severe"])
            mri_lacunes = st.selectbox("Lacunes", ["None","Mild","Moderate","Severe"])
            medial_temporal_atrophy = st.selectbox("Medial Temporal Atrophy", ["None","Mild","Moderate","Severe"])
            small_vessel_disease = st.selectbox("Small Vessel Disease", ["None","Mild","Moderate","Severe"])
           
            st.markdown("---")
            st.subheader("Quick Assessment Scores (0-100)")
            col1, col2 = st.columns(2)
            with col1:
                cog = st.slider("Cognitive Function", 0, 100,
                                st.session_state["doctor_tmp"].get("cognitive", 50))
                motor = st.slider("Motor Skills", 0, 100,
                                    st.session_state["doctor_tmp"].get("motor", 50))
            with col2:
                speech = st.slider("Speech Clarity", 0, 100,
                                       st.session_state["doctor_tmp"].get("speech", 50))
                mood = st.selectbox("Mood Status", ["Normal","Anxious","Depressed","Agitated"],
                                  index=["Normal","Anxious","Depressed","Agitated"].index(
                                      st.session_state["doctor_tmp"].get("mood", "Normal")))
           
            st.session_state["doctor_tmp"].update({
                "white_matter_lesions": white_matter_lesions, "mri_lacunes": mri_lacunes,
                "medial_temporal_atrophy": medial_temporal_atrophy, "small_vessel_disease": small_vessel_disease,
                "quick_scores": {"cognitive": int(cog), "motor": int(motor), "speech": int(speech), "mood": mood}
            })
       
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
       
        with col1:
            if current_section > 0:
                if st.button("← Previous Section", use_container_width=True):
                    st.session_state["assessment_section"] = current_section - 1
                    st.rerun()
            else:
                if st.button("Back to Facility", use_container_width=True):
                    st.session_state["page"] = "select_facility"
                    st.rerun()
       
        with col2:
            if current_section < total_sections - 1:
                if st.button("Next Section →", use_container_width=True, type="primary"):
                    st.session_state["assessment_section"] = current_section + 1
                    st.rerun()
            else:
                # Final section - Save and continue
                if st.button("Complete Assessment", use_container_width=True, type="primary"):
                    doc_data = st.session_state.get("doctor_tmp", {}).copy()
                    doc_data.update({
                        "assessing_doctor": user.get('assigned_doctor'),
                        "assessment_date": str(date.today())
                    })
                    users = load_users()
                    user_key = None
                    for key, u in users.items():
                        if u.get("patient_id") == user.get("patient_id"):
                            user_key = key
                            break
                   
                    if user_key:
                        users[user_key]["visits"][visit_index]["doctor_assessment"] = doc_data
                        save_users(users)
                        st.session_state["user"] = users[user_key]
                        st.session_state.pop("doctor_tmp", None)
                        st.session_state["assessment_section"] = 0  # Reset for next time
                        st.success("Clinical assessment completed successfully!")
                        st.session_state["page"] = "video_instructions"
                        st.rerun()
       
        with col3:
            # Section overview/summary
            completion_status = f"Section {current_section + 1}/{total_sections}"
            st.info(completion_status)

def page_video_instructions():
    st.title("Video Analysis Instructions")
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        st.markdown("""
        ### Video Assessment Phase - **60 Seconds**
       
        This assessment will analyze various neurological indicators through video recording.
       
        **Duration:** 60 seconds (1 minute)
       
        **What you'll do:**
        - Sit comfortably in front of your camera
        - Follow the on-screen instructions for each task
        - Each task changes automatically every 10 seconds
        - Ensure good lighting and visibility
       
        **6 Tasks (10 seconds each):**
        1. **0-10s:** Facial expressions & blinking assessment
        2. **10-20s:** Tremor observation (hands relaxed)
        3. **20-30s:** Finger tapping test
        4. **30-40s:** Walking and arm swing
        5. **40-50s:** Balance and postural stability
        6. **50-60s:** Facial symmetry & head tremor
        """)
   
    with col2:
        st.info("""
        **Preparation Tips:**
        - Ensure good lighting
        - Keep camera stable
        - Wear comfortable clothes
        - Have space to move around
        - Remove distracting background
        """)
       
        if "user" in st.session_state and st.session_state["user"]:
            user = st.session_state["user"]
            st.success(f"Patient: {user['name']}")
            st.info(f"ID: {user.get('patient_id', 'N/A')}")
   
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start Video Recording", use_container_width=True):
            st.session_state["page"] = "video_recording"
            st.rerun()
   
    with col_b:
        if st.button("Back to Assessment", use_container_width=True):
            st.session_state["page"] = "doctor_assessment"
            st.rerun()

def page_video_recording():
    st.title("Video Recording in Progress")
   
    left, right = st.columns([2, 3])
   
    with left:
        st.subheader("Current Task")
       
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
            st.session_state.recording_active = True
       
        # Auto-refresh every second
        st_autorefresh(interval=1000, key="video_timer_refresh")
       
        elapsed = int(time.time() - st.session_state.start_time) if st.session_state.recording_active else 0
        minutes, seconds = divmod(elapsed, 60)
       
        # Progress bar
        progress = min(elapsed / 60, 1.0)  # MODIFIED: 60 seconds total
        st.progress(progress)
        st.markdown(f"### Time: {minutes:02d}:{seconds:02d} / 01:00")  # MODIFIED: Show 1:00 max
       
        # Current task - MODIFIED: 6 tasks, 10 seconds each
        idx = min(elapsed // 10, len(FEATURE_GUIDELINES) - 1)
        feature, description = FEATURE_GUIDELINES[idx]
       
        st.markdown(f"**Task {idx + 1}/6:**")  # MODIFIED: Show 6 tasks
        st.markdown(f"### {feature}")
        st.info(description)
       
        if elapsed >= 60:  # MODIFIED: 60 seconds limit
            st.warning("Time limit reached! Please stop the recording.")
            st.balloons()
       
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Stop Early"):
                st.session_state.recording_active = False
        with col2:
            if st.button("Restart"):
                st.session_state.start_time = time.time()
                st.session_state.recording_active = True
                st.rerun()
   
    with right:
        st.subheader("Camera Feed")
        video_file = st.camera_input("Recording Active - Take photo when done", key="video_camera")
       
        if video_file:
            st.session_state.video_file = video_file
            st.session_state.recording_active = False
            st.success("Video recording complete!")
           
            if st.button("Analyze Video", use_container_width=True):
                st.session_state["page"] = "video_analysis"
                st.rerun()

def page_video_analysis():
    st.title("Video Analysis Results")
   
    if st.session_state.video_file is None:
        st.error("No video found. Please record again.")
        if st.button("Record Again"):
            st.session_state["page"] = "video_recording"
            st.rerun()
        return
   
    with st.spinner("Analyzing video... This may take a moment..."):
        time.sleep(2)  # Simulate processing time
       
        # Create mock frame analysis
        all_scores = []
        for i in range(6):  # Simulate 6 frames for 6 tasks
            all_scores.append(analyze_frame(None))
       
        # Average scores across frames
        avg_scores = {k: np.mean([f[k] for f in all_scores]) for k in all_scores[0].keys()}
        st.session_state.video_scores = avg_scores
       
        # Compute probabilities
        probs = compute_video_probabilities(avg_scores)
        st.session_state.video_probs = probs
   
    st.success("Video analysis complete!")
   
    # Display results
    col1, col2 = st.columns([1, 1])
   
    with col1:
        st.subheader("Disease Probability (Video)")
        fig_pie = px.pie(
            values=list(st.session_state.video_probs.values()),
            names=list(st.session_state.video_probs.keys()),
            title="Video-Based Prediction"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
   
    with col2:
        st.subheader("Feature Scores")
        fig_radar = create_radar_chart(st.session_state.video_scores, "Video Analysis")
        st.plotly_chart(fig_radar, use_container_width=True)
   
    # Detailed breakdown
    st.subheader("Detailed Feature Analysis")
   
    for feature, description in FEATURE_GUIDELINES:
        if feature in avg_scores:
            score = avg_scores[feature]
            col1, col2 = st.columns([2, 1])
           
            with col1:
                st.markdown(f"**{feature}**")
                st.caption(description)
           
            with col2:
                st.metric("Risk Score", f"{score:.2f}")
   
    # Continue to audio
    st.markdown("---")
    st.info("Next: Audio analysis for complete assessment.")
   
    if st.button("Continue to Audio Analysis", use_container_width=True):
        st.session_state["page"] = "audio_instructions"
        st.rerun()

def page_audio_instructions():
    st.title("Audio Analysis Instructions")
   
    col1, col2 = st.columns([2, 1])
   
    with col1:
        st.markdown("""
        ### Audio Assessment Phase - **60 Seconds**
       
        Record your speech for **60 seconds** to analyze speech patterns.
       
        **The system will guide you through 4 tasks:**
       
        1. **Reading (0-15s)**: Read the text shown clearly
        2. **Counting (15-30s)**: Count from 1 to 15
        3. **Word Recall (30-45s)**: Say three fruits you like
        4. **Description (45-60s)**: Describe your surroundings
       
        **Guidelines:**
        - Speak clearly at normal pace
        - Don't rush through tasks
        - Allow microphone access when prompted
        - Use a quiet environment
        """)
   
    with col2:
        st.info("""
        **Tips:**
        - Find a quiet room
        - Speak at normal volume
        - Follow on-screen prompts
        - Quality over speed
        """)
       
        if st.session_state.video_probs:
            st.success("Video analysis completed")
   
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start Audio Recording", use_container_width=True):
            st.session_state["page"] = "audio_recording"
            st.rerun()
   
    with col_b:
        if st.button("Back to Video Results", use_container_width=True):
            st.session_state["page"] = "video_analysis"
            st.rerun()

def page_audio_recording():
    st.title("Audio Recording")
   
    # Check if we already have processed audio
    if st.session_state.get("audio_processed", False):
        st.success("Audio recording and processing complete!")
       
        if st.session_state.get("audio_bytes"):
            st.audio(st.session_state.audio_bytes)
       
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Analysis Results", use_container_width=True, type="primary"):
                st.session_state["page"] = "audio_analysis"
                st.rerun()
       
        with col2:
            if st.button("Record Again", use_container_width=True):
                st.session_state.audio_processed = False
                st.session_state.streamlit_message = None
                if "audio_bytes" in st.session_state:
                    del st.session_state["audio_bytes"]
                st.rerun()
       
        if st.button("Back to Instructions"):
            st.session_state["page"] = "audio_instructions"
            st.rerun()
        return
   
    # Show recording interface
    st.markdown("### Record Your Speech")
    st.info("The recording will guide you through 4 tasks automatically.")
   
    # Audio recorder component
    comp_val = st.components.v1.html(AUDIO_RECORDER_HTML, height=350)
   
    # Handle audio data
    if comp_val and isinstance(comp_val, dict) and comp_val.get("type") == "AUDIO_DATA":
        st.session_state.streamlit_message = comp_val
   
    msg = st.session_state.get("streamlit_message")
    if msg and isinstance(msg, dict) and msg.get("type") == "AUDIO_DATA":
        b64 = msg.get("data")
        filename = msg.get("filename", f"recording_{int(time.time())}.webm")
       
        try:
            audio_bytes = base64.b64decode(b64)
            st.session_state.audio_bytes = audio_bytes
           
            audio_file_path = save_audio_file(audio_bytes, filename, "current_user")
            st.session_state.audio_file = audio_file_path
           
            st.success("Audio recording complete!")
            st.audio(audio_bytes)
           
            st.markdown("---")
            st.markdown("### Ready to Process Audio")
           
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Process & Analyze Audio", use_container_width=True, type="primary"):
                    st.session_state.audio_processed = True
                   
                    with st.spinner("Analyzing audio..."):
                        time.sleep(2)
                       
                        audio_scores = analyze_audio_simple()
                        st.session_state.audio_scores = audio_scores
                       
                        audio_probs = compute_audio_probabilities(audio_scores)
                        st.session_state.audio_probs = audio_probs
                   
                    st.success("Audio analysis complete!")
                    st.balloons()
                   
                    time.sleep(1)
                    st.session_state["page"] = "audio_analysis"
                    st.rerun()
           
            with col2:
                if st.button("Record Again", use_container_width=True):
                    st.session_state.streamlit_message = None
                    if "audio_bytes" in st.session_state:
                        del st.session_state["audio_bytes"]
                    st.rerun()
           
        except Exception as e:
            st.error(f"Error processing audio: {e}")
   
    else:
        st.markdown("### Recording Instructions")
        st.markdown("""
        **The recording will guide you through:**
       
        1. **0-15 seconds**: Read the text shown
        2. **15-30 seconds**: Count from 1 to 15
        3. **30-45 seconds**: Say three fruits
        4. **45-60 seconds**: Describe surroundings
        """)
   
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
   
    with col1:
        if st.button("Back to Instructions"):
            st.session_state["page"] = "audio_instructions"
            st.rerun()
   
    with col2:
        if st.button("Proceed to Analysis"):
            st.session_state["page"] = "audio_analysis"
            st.rerun()

def page_audio_analysis():
    st.title("Audio Analysis Results")
   
    # Generate analysis if not present
    if not st.session_state.audio_scores or not st.session_state.audio_probs:
        with st.spinner("Analyzing audio features..."):
            time.sleep(2)
           
            audio_scores = analyze_audio_simple()
            st.session_state.audio_scores = audio_scores
           
            audio_probs = compute_audio_probabilities(audio_scores)
            st.session_state.audio_probs = audio_probs
   
    st.success("Audio analysis complete!")
   
    # Display results
    col1, col2 = st.columns([1, 1])
   
    with col1:
        st.subheader("Disease Probability (Audio)")
        fig_pie = px.pie(
            values=list(st.session_state.audio_probs.values()),
            names=list(st.session_state.audio_probs.keys()),
            title="Audio-Based Prediction"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
   
    with col2:
        st.subheader("Feature Scores")
        fig_radar = create_radar_chart(st.session_state.audio_scores, "Audio Analysis")
        st.plotly_chart(fig_radar, use_container_width=True)
   
    # Detailed breakdown
    st.subheader("Detailed Feature Analysis")
   
    for feature, description in AUDIO_FEATURES:
        if feature in st.session_state.audio_scores:
            score = st.session_state.audio_scores[feature]
            col1, col2 = st.columns([2, 1])
           
            with col1:
                st.markdown(f"**{feature}**")
                st.caption(description)
           
            with col2:
                st.metric("Risk Score", f"{score:.2f}")
   
    # Continue to final results
    st.markdown("---")
    st.info("Ready for combined analysis results.")
   
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Final Results", use_container_width=True, type="primary"):
            st.session_state["page"] = "final_results"
            st.rerun()
   
    with col2:
        if st.button("Back to Audio Recording", use_container_width=True):
            st.session_state["page"] = "audio_recording"
            st.rerun()

def page_final_results():
    st.title("Complete NeuroHealth Assessment")
   
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        visit_index = st.session_state.get("current_visit_index", -1)
       
        st.markdown(f"### Patient: {user['name']}, ID: {user.get('patient_id', 'N/A')}, Age: {user.get('age', 'N/A')}")
        st.markdown(f"### Blood Group: {user.get('blood_group', 'N/A')}, Doctor: {user.get('assigned_doctor', 'N/A')}")
       
        if visit_index >= 0 and visit_index < len(user.get("visits", [])):
            visit = user["visits"][visit_index]
            st.markdown(f"**Visit Date:** {visit.get('date')} | **Doctor:** {visit.get('doctor', 'N/A')}")
   
    if not st.session_state.video_probs or not st.session_state.audio_probs:
        st.error("Missing analysis data. Please complete both assessments.")
        return
   
    # Display Clinical Assessment First
    if "user" in st.session_state and visit_index >= 0:
        visit = user.get("visits", [])[visit_index] if visit_index < len(user.get("visits", [])) else {}
        doctor_assessment = visit.get("doctor_assessment")
       
        if doctor_assessment:
            st.subheader("Clinical Assessment Summary")
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                st.markdown("#### Test Scores")
                st.write(f"**MMSE Score:** {doctor_assessment.get('mmse', 'N/A')}/30")
                st.write(f"**MoCA Score:** {doctor_assessment.get('moca', 'N/A')}/30")
                st.write(f"**UPDRS Score:** {doctor_assessment.get('updrs', 'N/A')}")
                st.write(f"**NIHSS Score:** {doctor_assessment.get('nihss', 'N/A')}")
               
                st.markdown("#### Quick Assessment")
                qs = doctor_assessment.get("quick_scores", {})
                st.write(f"**Cognitive:** {qs.get('cognitive', 'N/A')}/100")
                st.write(f"**Motor:** {qs.get('motor', 'N/A')}/100")
                st.write(f"**Speech:** {qs.get('speech', 'N/A')}/100")
                st.write(f"**Mood:** {qs.get('mood', 'N/A')}")
           
            with col2:
                st.markdown("#### Key Clinical Observations")
                st.write(f"**Memory Loss:** {doctor_assessment.get('memory', 'N/A')}")
                st.write(f"**Tremors:** {doctor_assessment.get('tremors', 'N/A')}")
                st.write(f"**Rigidity:** {doctor_assessment.get('rigidity', 'N/A')}")
                st.write(f"**Gait Issues:** {doctor_assessment.get('gait', 'N/A')}")
                st.write(f"**Speech Issues:** {doctor_assessment.get('speech_issues', 'N/A')}")
                st.write(f"**Facial Expression:** {doctor_assessment.get('facial', 'N/A')}")
                st.write(f"**Handwriting Issues:** {doctor_assessment.get('handwriting', 'N/A')}")
                st.write(f"**Balance Issues:** {doctor_assessment.get('hemiparesis', 'N/A')}")
           
            with col3:
                st.markdown("#### Medical History")
                st.write(f"**Hypertension:** {doctor_assessment.get('htn', 'N/A')}")
                st.write(f"**Diabetes:** {doctor_assessment.get('diabetes', 'N/A')}")
                st.write(f"**Prior Stroke/TIA:** {doctor_assessment.get('prior_tia_stroke', 'N/A')}")
                st.write(f"**Head Injury:** {doctor_assessment.get('head_injury', 'N/A')}")
               
                st.markdown("#### MRI Findings")
                st.write(f"**White Matter Lesions:** {doctor_assessment.get('white_matter_lesions', 'N/A')}")
                st.write(f"**Temporal Atrophy:** {doctor_assessment.get('medial_temporal_atrophy', 'N/A')}")
                st.write(f"**Small Vessel Disease:** {doctor_assessment.get('small_vessel_disease', 'N/A')}")
           
            st.markdown("---")
   
    # Combine predictions
    combined_probs = combine_predictions(st.session_state.video_probs, st.session_state.audio_probs)
   
    # AI Analysis Results
    st.subheader("AI Analysis Results (Multi-Modal)")
   
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.markdown("#### Video Analysis")
        fig_video = px.pie(
            values=list(st.session_state.video_probs.values()),
            names=list(st.session_state.video_probs.keys()),
            title="Video-Based Prediction"
        )
        st.plotly_chart(fig_video, use_container_width=True)
   
    with col2:
        st.markdown("#### Audio Analysis")
        fig_audio = px.pie(
            values=list(st.session_state.audio_probs.values()),
            names=list(st.session_state.audio_probs.keys()),
            title="Audio-Based Prediction"
        )
        st.plotly_chart(fig_audio, use_container_width=True)
   
    with col3:
        st.markdown("#### Combined AI Prediction")
        fig_combined = px.pie(
            values=list(combined_probs.values()),
            names=list(combined_probs.keys()),
            title="Final AI Prediction"
        )
        st.plotly_chart(fig_combined, use_container_width=True)
   
    # Detailed comparison table
    st.subheader("Analysis Comparison Table")
   
    comparison_df = pd.DataFrame({
        'Condition': list(combined_probs.keys()),
        'Video Analysis': [st.session_state.video_probs.get(k, 0) for k in combined_probs.keys()],
        'Audio Analysis': [st.session_state.audio_probs.get(k, 0) for k in combined_probs.keys()],
        'Combined AI Result': list(combined_probs.values())
    })
   
    st.dataframe(comparison_df, use_container_width=True)
   
    # Get highest predictions
    highest_ai_risk = max(combined_probs, key=combined_probs.get)
    highest_ai_prob = combined_probs[highest_ai_risk]
   
    # Clinical recommendations
    st.subheader("Clinical Recommendations")
   
    recommendations = []
   
    if doctor_assessment:
        # Based on clinical findings
        if doctor_assessment.get('mmse', 30) < 24 or doctor_assessment.get('moca', 30) < 26:
            recommendations.append("Comprehensive neuropsychological testing")
       
        if doctor_assessment.get('updrs', 0) > 20 or doctor_assessment.get('tremors') in ['Moderate', 'Severe']:
            recommendations.append("Movement disorder specialist consultation")
       
        if doctor_assessment.get('nihss', 0) > 0 or doctor_assessment.get('prior_tia_stroke') == 'Yes':
            recommendations.append("Vascular neurology evaluation and stroke prevention")
       
        # Based on AI predictions - adjusted thresholds for better normal detection
        if highest_ai_risk == "Normal" and highest_ai_prob > 0.5:
            st.success("✅ **Analysis suggests normal neurological function**")
            st.info("Continue regular health monitoring and annual check-ups.")
        elif highest_ai_risk == "Parkinson's" and highest_ai_prob > 0.35:
            recommendations.append("Consider DaTscan imaging for Parkinson's disease evaluation")
        elif highest_ai_risk == "Alzheimer's" and highest_ai_prob > 0.35:
            recommendations.append("Alzheimer's disease biomarker testing")
        elif highest_ai_risk == "Stroke" and highest_ai_prob > 0.35:
            recommendations.append("Comprehensive stroke workup and prevention strategies")
        elif highest_ai_risk == "Brain Tumor" and highest_ai_prob > 0.25:
            recommendations.append("Brain MRI with contrast for tumor evaluation")
       
        if recommendations:
            st.markdown("**Recommended Follow-up Actions:**")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        elif highest_ai_risk != "Normal":
            st.info("Continue monitoring. Consider follow-up if symptoms persist or worsen.")
    else:
        # AI-only recommendations
        if highest_ai_risk == "Normal" and highest_ai_prob > 0.5:
            st.success("✅ **Analysis suggests normal neurological function**")
            st.info("Continue regular health monitoring and maintain healthy lifestyle.")
        elif highest_ai_prob > 0.4 and highest_ai_risk != "Normal":
            st.warning(f"⚠️ **Elevated risk detected: {highest_ai_risk}** ({highest_ai_prob*100:.1f}%)")
            st.markdown(f"""
            **Recommended Actions for {highest_ai_risk}:**
            - Consult neurologist for professional evaluation
            - Consider additional diagnostic tests
            - Monitor symptoms and seek attention if worsening
            - Remember: This is a screening tool, not definitive diagnosis
            """)
        else:
            st.info("Results show mixed indicators. Consult healthcare professional if you have concerns.")
   
    # Hospital Recommendations for Higher Opinion
    st.subheader("Recommendation for Higher Opinion")
   
    # Determine if higher opinion is needed
    need_higher_opinion = False
    reasons_for_referral = []
   
    if doctor_assessment:
        if doctor_assessment.get('mmse', 30) < 20 or doctor_assessment.get('moca', 30) < 20:
            need_higher_opinion = True
            reasons_for_referral.append("Severe cognitive impairment detected")
       
        if doctor_assessment.get('updrs', 0) > 30:
            need_higher_opinion = True
            reasons_for_referral.append("Significant motor dysfunction")
       
        if doctor_assessment.get('nihss', 0) > 5:
            need_higher_opinion = True
            reasons_for_referral.append("Stroke-related complications")
   
    # Adjusted AI thresholds - higher bar for referral to reduce false positives
    if highest_ai_prob > 0.6 and highest_ai_risk != "Normal":
        need_higher_opinion = True
        reasons_for_referral.append(f"High AI confidence for {highest_ai_risk} ({highest_ai_prob*100:.1f}%)")
    elif highest_ai_prob > 0.4 and highest_ai_risk in ["Brain Tumor", "Stroke"]:
        need_higher_opinion = True
        reasons_for_referral.append(f"Concerning findings for {highest_ai_risk} requiring urgent evaluation")
   
    if need_higher_opinion:
        st.warning("**Higher Opinion Recommended**")
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            st.markdown("#### Reasons for Referral:")
            for reason in reasons_for_referral:
                st.write(f"• {reason}")
           
            # Hospital selection
            st.markdown("#### Select Specialist Hospital:")
            hospital_options = ["Select Hospital"] + list(REFERRAL_HOSPITALS.keys())
            selected_hospital = st.selectbox("Choose Hospital", hospital_options)
           
            if selected_hospital != "Select Hospital":
                hospital_info = REFERRAL_HOSPITALS[selected_hospital]
               
                # Display hospital info
                st.info(f"**Speciality:** {hospital_info['speciality']}")
                st.info(f"**Contact:** {hospital_info['contact']}")
               
                # Doctor selection
                doctor_options = ["Select Doctor"] + hospital_info["doctors"]
                selected_doctor = st.selectbox("Choose Specialist Doctor", doctor_options)
               
                if selected_doctor != "Select Doctor":
                    st.success(f"Referral to {selected_hospital} - {selected_doctor}")
       
        with col2:
            st.info("**Next Steps**")
            st.write("1. Select hospital and doctor")
            st.write("2. Schedule appointment")
            st.write("3. Bring all test results")
    else:
        st.success("**No Higher Opinion Required at This Time**")
        if highest_ai_risk == "Normal" and highest_ai_prob > 0.5:
            st.info("**Analysis indicates normal neurological function.** Continue regular health monitoring and annual check-ups at the Primary Health Care Center.")
        else:
            st.info("Continue regular monitoring and follow-up care at the Primary Health Care Center. Schedule follow-up if symptoms develop or worsen.")
   
    # Save results to user visit
    if "user" in st.session_state and st.session_state["user"] and visit_index >= 0:
        users = load_users()
        user_key = None
        for key, u in users.items():
            if u.get("patient_id") == user.get("patient_id"):
                user_key = key
                break
       
        if user_key and visit_index < len(users[user_key]["visits"]):
            analysis_results = {
                "video_scores": st.session_state.video_scores,
                "video_probs": st.session_state.video_probs,
                "audio_scores": st.session_state.audio_scores,
                "audio_probs": st.session_state.audio_probs,
                "combined_probs": combined_probs,
                "analysis_date": str(date.today())
            }
            users[user_key]["visits"][visit_index]["multimodal_analysis"] = analysis_results
            save_users(users)
            st.session_state["user"] = users[user_key]
   
    st.markdown("---")
   
    # Navigation options
    col1, col2, col3 = st.columns(3)
   
    with col1:
        if st.button("New Assessment", use_container_width=True):
            # Reset analysis variables
            for key in ["video_file", "video_scores", "video_probs", "audio_file",
                        "audio_scores", "audio_probs", "start_time", "recording_active",
                        "streamlit_message", "audio_processed", "audio_bytes", "assessment_section"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["page"] = "home"
            st.rerun()
   
    with col2:
        if st.button("Export Results", use_container_width=True):
            results_data = {
                "patient_info": {
                    "name": user.get("name") if "user" in st.session_state else "Unknown",
                    "patient_id": user.get("patient_id") if "user" in st.session_state else "N/A",
                    "age": user.get("age") if "user" in st.session_state else "N/A",
                    "assessment_date": str(date.today())
                },
                "combined_results": combined_probs,
                "recommendation": {
                    "primary_concern": highest_ai_risk,
                    "confidence": highest_ai_prob
                }
            }
            st.download_button(
                label="Download JSON Report",
                data=json.dumps(results_data, indent=2),
                file_name=f"neurohealth_report_{user.get('patient_id', 'unknown')}_{date.today()}.json",
                mime="application/json"
            )
   
    with col3:
        if st.button("Back to Home", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()
   
    # Medical disclaimer
    st.markdown("---")
    st.warning("""
    **Medical Disclaimer**: This application is a screening tool for research and educational
    purposes only. It is not intended to replace professional medical diagnosis, treatment,
    or advice. Always consult with qualified healthcare professionals for proper medical
    evaluation and treatment decisions.
    """)

# ------------------------
# SIDEBAR AND MAIN ROUTER
# ------------------------

def render_sidebar():
    with st.sidebar:
        st.title("NeuroHealth System")
       
        # Doctor info
        if "doctor" in st.session_state and st.session_state["doctor"]:
            doctor = st.session_state["doctor"]
            st.success(f"Doctor: {doctor['name']}")
           
        # User info
        if "user" in st.session_state and st.session_state["user"]:
            user = st.session_state["user"]
            st.success(f"Patient: {user['name']}")
            st.info(f"ID: {user.get('patient_id', 'N/A')}")
            st.info(f"Doctor: {user.get('assigned_doctor', 'N/A')}")
           
            # Current visit info
            visit_index = st.session_state.get("current_visit_index", -1)
            if visit_index >= 0 and visit_index < len(user.get("visits", [])):
                visit = user["visits"][visit_index]
                st.write(f"**Current Visit:** {visit.get('date')}")
               
                # Assessment progress indicator
                if st.session_state.get("page") == "doctor_assessment":
                    current_section = st.session_state.get("assessment_section", 0)
                    st.write(f"**Assessment:** Section {current_section + 1}/5")

def main():
    render_sidebar()
   
    # Page routing
    page_functions = {
        "patient_register": page_patient_register,
        "doctor_login": page_doctor_login,
        "doctor_dashboard": page_doctor_dashboard,
        "patient_register_by_doctor": page_patient_register_by_doctor,
        "home": page_home,
        "patient_info": page_patient_info,
        "visiting_data": page_visiting_data,
        "select_facility": page_select_facility,
        "doctor_assessment": page_doctor_assessment,
        "video_instructions": page_video_instructions,
        "video_recording": page_video_recording,
        "video_analysis": page_video_analysis,
        "audio_instructions": page_audio_instructions,
        "audio_recording": page_audio_recording,
        "audio_analysis": page_audio_analysis,
        "final_results": page_final_results
    }
   
    current_page = st.session_state.get("page", "patient_register")
   
    if current_page in page_functions:
        page_functions[current_page]()
    else:
        st.error("Page not found!")
        st.session_state["page"] = "patient_register"
        st.rerun()

# Run the application
if __name__ == "__main__":
    main()
