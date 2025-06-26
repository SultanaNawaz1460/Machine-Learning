"""
Face Recognition System - Enhanced Real-time Recognition
This script performs real-time face recognition with multiple models and improved performance.
"""
import os
import cv2
import pickle
import numpy as np
import sqlite3
import time
import argparse

def load_face_recognition_models():
    """Load all available face recognition models"""
    models = {}
    label_names = {}
    
    # Check if model directory exists
    if not os.path.exists("model"):
        print("Error: Model directory not found! Please train models first.")
        return None
    
    # Load the label mapping (ID to name)
    if not os.path.exists("model/face_labels.pickle"):
        print("Error: Face labels file not found! Please train the model first.")
        return None
    
    try:
        with open("model/face_labels.pickle", "rb") as f:
            label_names = pickle.load(f)
        
        print(f"Label mapping loaded with {len(label_names)} persons:")
        for person_id, name in label_names.items():
            print(f"  {person_id}: {name}")
    except Exception as e:
        print(f"Error loading label mapping: {str(e)}")
        return None
    
    # Try to load all model types
    model_files = {
        "eigenfaces": "model/face_model_eigenfaces.yml",
        "fisherfaces": "model/face_model_fisherfaces.yml"
    }
    
    for model_type, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                recognizer = None
                if model_type == "eigenfaces":
                    recognizer = cv2.face.EigenFaceRecognizer_create()
                elif model_type == "fisherfaces":
                    recognizer = cv2.face.FisherFaceRecognizer_create()
                
                recognizer.read(model_path)
                models[model_type] = recognizer
                print(f"Loaded {model_type.upper()} model from {model_path}")
            except Exception as e:
                print(f"Error loading {model_type} model: {str(e)}")
    
    if not models:
        print("No models were loaded successfully. Please retrain the models.")
        return None
    
    return {
        "models": models,
        "label_names": label_names
    }

def get_person_details(person_id):
    """Get person details from database by ID"""
    conn = sqlite3.connect('face_recognition_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, age, gender, department, semester, registration_number 
        FROM persons WHERE id = ?
    """, (person_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "name": result[0],
            "age": result[1],
            "gender": result[2],
            "department": result[3],
            "semester": result[4],
            "registration_number": result[5]
        }
    return None

def run_face_recognition(model_type="eigenfaces", confidence_threshold=None):
    print("\n=== Enhanced Real-time Face Recognition System ===")
    
    # Set default thresholds based on model type
    default_thresholds = {
        "eigenfaces": 5000,   # Higher is better match for Eigenfaces
        "fisherfaces": 3000   # Higher is better match for Fisherfaces
    }
    
    if confidence_threshold is None:
        confidence_threshold = default_thresholds.get(model_type, 5000)
    
    print(f"Using model: {model_type.upper()}")
    print(f"Initial confidence threshold: {confidence_threshold}")
    
    # Load the trained models
    model_data = load_face_recognition_models()
    if model_data is None:
        return
    
    # Check if selected model is available
    if model_type not in model_data["models"]:
        print(f"Error: {model_type.upper()} model not found. Available models:")
        for available_model in model_data["models"].keys():
            print(f"  - {available_model}")
        return
    
    face_recognizer = model_data["models"][model_type]
    label_names = model_data["label_names"]
    
    if not label_names:
        print("No face labels found in the model. Please train the model first.")
        return
    
    # Load OpenCV's pre-trained face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_detector.empty():
        print("Error: Failed to load face detector cascade.")
        return
    
    # Initialize webcam
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Get camera frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Starting real-time face recognition...")
    print("Press 'q' to quit")
    print("Press 'd' to toggle debug mode")
    print("Press 'm' to switch model")
    print("Press '+' to increase threshold")
    print("Press '-' to decrease threshold")
    
    # For frame rate calculation
    frame_time = time.time()
    frame_count = 0
    
    # Debug mode flag
    debug_mode = True
    
    # For ensemble prediction
    use_ensemble = False
    
    # Process every frame
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:
            fps = frame_count / (time.time() - frame_time)
            frame_time = time.time()
            frame_count = 0
        else:
            fps = 0
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Detect faces in the frame
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each face found in the frame
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to a standard size as used during training
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Additional preprocessing
            face_roi = cv2.equalizeHist(face_roi)  # Enhance contrast
            
            try:
                # Predict the face
                person_id = None
                confidence = None
                name = "Unknown"
                person_details = None
                
                if use_ensemble and len(model_data["models"]) > 1:
                    # Ensemble prediction (use all available models and vote)
                    predictions = {}
                    
                    for model_name, model in model_data["models"].items():
                        pred_id, pred_conf = model.predict(face_roi)
                        
                        # Check if prediction passes threshold
                        if pred_conf > default_thresholds[model_name]:
                            if pred_id not in predictions:
                                predictions[pred_id] = 0
                            predictions[pred_id] += 1
                    
                    # Get the most voted prediction
                    if predictions:
                        person_id = max(predictions, key=predictions.get)
                        confidence = predictions[person_id] / len(model_data["models"])
                else:
                    # Use selected model
                    person_id, confidence = face_recognizer.predict(face_roi)
                
                # Check against confidence threshold - higher confidence is better for Eigenfaces/Fisherfaces
                threshold_passed = confidence > confidence_threshold
                
                if threshold_passed and person_id in label_names:
                    name = label_names[person_id]
                    
                    # Get person details from database
                    person_details = get_person_details(person_id)
                    if person_details:
                        name = person_details["name"]
                
                # Draw rectangle around face
                face_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), face_color, 2)
                
                # Draw name label
                cv2.rectangle(display_frame, (x, y+h), (x+w, y+h+35), face_color, cv2.FILLED)
                cv2.putText(display_frame, f"{name} ({confidence:.1f})", (x+6, y+h+25), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Display person details if recognized
                if person_details:
                    # Create info text
                    info_text = [
                        f"Name: {person_details['name']}",
                        f"Age: {person_details['age']}",
                        f"Gender: {person_details['gender']}",
                        f"Department: {person_details['department']}",
                        f"Semester: {person_details['semester']}",
                        f"Reg#: {person_details['registration_number']}"
                    ]
                    
                    # Calculate info box position
                    info_x = max(10, x - 50)
                    info_y = max(y - 150, 50)
                    
                    # Draw info background
                    text_height = len(info_text) * 25
                    cv2.rectangle(display_frame, (info_x - 10, info_y - 30), 
                                 (info_x + 300, info_y + text_height), 
                                 (0, 0, 0), cv2.FILLED)
                    
                    # Draw info text
                    for i, line in enumerate(info_text):
                        cv2.putText(display_frame, line, (info_x, info_y + i * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Debug mode display
                if debug_mode and name == "Unknown":
                    # Show confidence scores for all registered persons
                    debug_y = 60
                    cv2.putText(display_frame, "Debug - Confidence Scores:", (10, debug_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    debug_y += 25
                    
                    # Calculate and display confidence for each person
                    max_debug_persons = 3  # Limit to top 3 for clarity
                    debug_scores = []
                    
                    for pid in label_names:
                        p_name = label_names[pid]
                        p_id, p_confidence = face_recognizer.predict(face_roi)
                        debug_scores.append((p_name, p_confidence))
                    
                    # Sort by confidence (higher is better for Eigenfaces/Fisherfaces)
                    debug_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Display top matching persons
                    for i, (p_name, p_conf) in enumerate(debug_scores[:max_debug_persons]):
                        cv2.putText(display_frame, f"{p_name}: {p_conf:.1f}", 
                                   (10, debug_y + i * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        
            except Exception as e:
                print(f"Recognition error: {str(e)}")
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(display_frame, "Error", (x+6, y+h+25), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display FPS counter
        if fps > 0:
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display model info
        cv2.putText(display_frame, f"Model: {model_type.upper()}", 
                   (frame_width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display threshold info
        threshold_text = f"Threshold: {confidence_threshold} (higher is better)"
        cv2.putText(display_frame, threshold_text, 
                   (frame_width - 300, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display debug mode status
        mode_text = "Debug: ON" if debug_mode else "Debug: OFF"
        cv2.putText(display_frame, mode_text, (frame_width - 120, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Face Recognition', display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('e'):
            use_ensemble = not use_ensemble
            print(f"Ensemble prediction: {'ON' if use_ensemble else 'OFF'}")
        elif key == ord('m'):
            # Cycle through available models
            models_list = list(model_data["models"].keys())
            current_idx = models_list.index(model_type)
            next_idx = (current_idx + 1) % len(models_list)
            model_type = models_list[next_idx]
            face_recognizer = model_data["models"][model_type]
            confidence_threshold = default_thresholds[model_type]
            print(f"Switched to {model_type.upper()} model with threshold {confidence_threshold}")
        elif key in [ord('+'), ord('=')]:
            confidence_threshold += 500  # For Eigenfaces/Fisherfaces, higher threshold = stricter
            print(f"Adjusted threshold to: {confidence_threshold}")
        elif key == ord('-'):
            confidence_threshold -= 500  # For Eigenfaces/Fisherfaces, lower threshold = more lenient
            print(f"Adjusted threshold to: {confidence_threshold}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enhanced Face Recognition System')
    parser.add_argument('--model', type=str, default='eigenfaces', choices=['eigenfaces', 'fisherfaces'],
                        help='Face recognition model to use')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Confidence threshold for recognition')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run face recognition
    run_face_recognition(model_type=args.model, confidence_threshold=args.threshold)

if __name__ == "__main__":
    main()