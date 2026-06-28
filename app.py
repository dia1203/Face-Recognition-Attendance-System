"""
Facial Recognition Attendance System with Liveness Detection
Built with Streamlit, OpenCV, Dlib, and Face Recognition.
"""

import streamlit as st 
import cv2
import numpy as np
import pandas as pd
import os
import datetime
from pathlib import Path
import face_recognition
import time
import dlib
from scipy.spatial import distance as dist
from PIL import Image

# Import your custom modules
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.attendance_manager import AttendanceManager
from utils.config import Config
from utils.database import Database
from utils.helpers import setup_directories, log_activity

# Streamlit page setup
st.set_page_config(
    page_title="Facial Recognition Attendance System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class AttendanceApp:
    def __init__(self):
        self.config = Config()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.attendance_manager = AttendanceManager()
        self.database = Database()

        setup_directories()

        if "attendance_marked" not in st.session_state:
            st.session_state.attendance_marked = []

    # ------------------------------
    # MAIN RUN FUNCTION
    # ------------------------------
    def run(self):
        st.markdown('<h1 class="main-header">🎯 Facial Recognition Attendance System</h1>',
                    unsafe_allow_html=True)

        page = st.sidebar.selectbox(
            "Select Mode",
            ["📸 Live Attendance", "👥 Person Management", "📊 Reports", "⚙️ Settings"]
        )

        if page == "📸 Live Attendance":
            self.live_attendance_page()
        elif page == "👥 Person Management":
            self.person_management_page()
        elif page == "📊 Reports":
            self.reports_page()
        elif page == "⚙️ Settings":
            self.settings_page()

    # ------------------------------
    # LIVE ATTENDANCE
    # ------------------------------
    def live_attendance_page(self):
        st.subheader("📸 Live Attendance Tracking")

        col1, col2 = st.columns([2, 1])

        with col1:
            camera_source = st.selectbox("Select Camera Source", [0, 1, 2])
            confidence_threshold = st.slider(
                "Recognition Confidence", 0.3, 0.7, 0.4)
            if st.button("Start Attendance Tracking", type="primary"):
                self.run_live_tracking(camera_source, confidence_threshold)

        with col2:
            self.display_today_stats()

    def run_live_tracking(self, camera_source, confidence_threshold):
        cap = cv2.VideoCapture(camera_source)
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        stop_button = st.button("Stop Tracking")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break

            faces = self.face_detector.detect_faces(frame)
            recognized_persons = []

            for face_location in faces:
                face_encoding = face_recognition.face_encodings(
                    frame, [face_location])

                if face_encoding:
                    person_name, confidence = self.face_recognizer.recognize_face(
                        face_encoding[0], confidence_threshold
                    )

                    if person_name != "Unknown":
                        recognized_persons.append(person_name)

                        # --- LIVENESS DETECTION ---
                        st.write(
                            f"Please blink to confirm your presence, {person_name}...")
                        blink_detected = self.check_blink(frame)

                        if blink_detected:
                            attendance_status = self.attendance_manager.mark_attendance(
                                person_name)
                            st.success(
                                f"✅ Attendance marked for {person_name}")
                            top, right, bottom, left = face_location
                            cv2.rectangle(
                                frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, f"{person_name} ({confidence:.2f})",
                                        (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            st.warning(
                                f"⚠️ No blink detected for {person_name} — attendance not marked.")
                    else:
                        # Unknown face
                        top, right, bottom, left = face_location
                        cv2.rectangle(
                            frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            if recognized_persons:
                status_placeholder.success(
                    f"Recognized: {', '.join(recognized_persons)}")
            else:
                status_placeholder.info("Looking for faces...")

            time.sleep(0.1)

        cap.release()

    # ------------------------------
    # BLINK / LIVENESS DETECTION
    # ------------------------------
    def check_blink(self, frame):
        """Detect blink using eye aspect ratio (EAR)"""
        LANDMARK_PATH = "models/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(LANDMARK_PATH):
            st.error("Missing shape_predictor_68_face_landmarks.dat in /models folder")
            return False

        predictor = dlib.shape_predictor(LANDMARK_PATH)
        detector = dlib.get_frontal_face_detector()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        def eye_aspect_ratio(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.20:  # Blink threshold
                return True
        return False

    # ------------------------------
    # PERSON MANAGEMENT
    # ------------------------------
    def person_management_page(self):
        st.subheader("👥 Person Management")
        tab1, tab2, tab3 = st.tabs(["Add Person", "View People", "Remove Person"])

        with tab1:
            self.add_person_form()
        with tab2:
            self.view_people()
        with tab3:
            self.remove_person_form()

    def add_person_form(self):
        st.markdown("### Add New Person")
        person_name = st.text_input("Person Name")
        person_id = st.text_input("Person ID (Optional)")
        uploaded_images = st.file_uploader(
            "Upload Person Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

        if st.button("Add Person") and person_name and uploaded_images:
            try:
                person_dir = Path(f"data/faces/{person_name}")
                person_dir.mkdir(parents=True, exist_ok=True)
                face_encodings = []
                for i, uploaded_file in enumerate(uploaded_images):
                    image_path = person_dir / f"{person_name}_{i}.jpg"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        face_encodings.extend(encodings)

                if face_encodings:
                    self.face_recognizer.add_person(person_name, face_encodings)
                    self.database.add_person(person_name, person_id)
                    st.success(f"Added {person_name} with {len(face_encodings)} encodings!")
                else:
                    st.error("No faces detected in uploaded images")
            except Exception as e:
                st.error(f"Error adding person: {str(e)}")

    def view_people(self):
        st.markdown("### Registered People")
        people = self.database.get_all_people()
        if people:
            df = pd.DataFrame(people, columns=['Name', 'ID', 'Date Added'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No people registered yet.")

    def remove_person_form(self):
        st.markdown("### Remove Person")
        people = self.database.get_all_people()
        if people:
            names = [p[0] for p in people]
            selected = st.selectbox("Select Person to Remove", names)
            if st.button("Remove Person", type="secondary"):
                import shutil
                try:
                    self.face_recognizer.remove_person(selected)
                    self.database.remove_person(selected)
                    dir_path = Path(f"data/faces/{selected}")
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                    st.success(f"Removed {selected}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("No registered people found.")

    # ------------------------------
    # REPORTS
    # ------------------------------
    def reports_page(self):
        st.subheader("📊 Attendance Reports")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.date.today())

        if st.button("Generate Report"):
            data = self.attendance_manager.get_attendance_report(start_date, end_date)
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "attendance_report.csv", "text/csv")
            else:
                st.info("No data found for selected period.")

    # ------------------------------
    # SETTINGS
    # ------------------------------
    def settings_page(self):
        st.subheader("⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Detection Method", ["HOG", "CNN"])
            tolerance = st.slider("Recognition Tolerance", 0.3, 0.7, 0.4)
        with col2:
            st.info(f"OpenCV: {cv2.__version__}")
            st.info(f"Registered People: {len(self.database.get_all_people())}")

    # ------------------------------
    # TODAY'S STATS
    # ------------------------------
    def display_today_stats(self):
        st.markdown("### Today's Statistics")
        today = self.attendance_manager.get_today_attendance()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Present Today", len(today))
        with col2:
            total = len(self.database.get_all_people())
            st.metric("Total Registered", total)

        if today:
            st.markdown("### Recent Check-ins")
            for record in today[-5:]:
                st.text(f"{record['name']} - {record['time']}")


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app = AttendanceApp()
    app.run()
