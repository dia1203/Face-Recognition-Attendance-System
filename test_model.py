import dlib

try:
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    print("✅ Model loaded successfully! Everything is set up correctly.")
except Exception as e:
    print("❌ Error loading model:", e)
