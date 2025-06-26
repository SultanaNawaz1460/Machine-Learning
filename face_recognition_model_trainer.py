"""
Face Recognition System - Data Relabeling and Model Retraining
This script reorganizes existing face data and retrains the model with correct labels.
LBPH model has been removed, using only Eigenfaces and Fisherfaces.
"""
import os
import shutil
import cv2
import numpy as np
import sqlite3
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# STEP 1: Database checking and verification
def check_database():
    """Check if database exists and contains persons"""
    print("\n=== Checking Database ===")
    if not os.path.exists('face_recognition_db.sqlite'):
        print("Database not found! Please run the database setup script first.")
        return False
    
    conn = sqlite3.connect('face_recognition_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM persons")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("No persons found in the database! Please add persons first.")
        conn.close()
        return False
    
    print(f"Database found with {count} persons registered.")
    
    # Show all registered persons
    df = pd.read_sql_query("SELECT id, name, registration_number, image_folder FROM persons", conn)
    print("\nRegistered persons:")
    print(df)
    conn.close()
    
    return True

# STEP 2: Verify each person's image folder
def verify_image_folders():
    """Check if image folders exist and have images"""
    print("\n=== Verifying Image Folders ===")
    conn = sqlite3.connect('face_recognition_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, image_folder FROM persons")
    persons = cursor.fetchall()
    conn.close()
    
    valid_persons = []
    for person_id, name, folder in persons:
        if not os.path.exists(folder):
            print(f"Warning: Folder for {name} does not exist: {folder}")
            continue
        
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"Warning: No images found for {name} in {folder}")
            continue
        
        print(f"Found {len(image_files)} images for {name}")
        valid_persons.append((person_id, name, folder, len(image_files)))
    
    return valid_persons

# STEP 3: Reorganize data with proper labeling
def reorganize_data(valid_persons):
    """Reorganize images into a clean structure with proper labeling"""
    print("\n=== Reorganizing Data ===")
    
    # Create a fresh data directory
    data_dir = "processed_data"
    if os.path.exists(data_dir):
        response = input(f"Directory {data_dir} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return None
        shutil.rmtree(data_dir)
    
    os.makedirs(data_dir)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Failed to load face detector cascade.")
        return None
    
    # Process each person's images
    all_face_data = []
    all_face_labels = []
    label_names = {}
    skipped_count = 0
    
    for person_id, name, folder, img_count in valid_persons:
        print(f"\nProcessing images for {name}...")
        person_dir = os.path.join(data_dir, f"{person_id}_{name.replace(' ', '_')}")
        os.makedirs(person_dir)
        
        # Add to label mapping
        label_names[person_id] = name
        
        # Process each image
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        processed_count = 0
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(folder, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    skipped_count += 1
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) != 1:
                    skipped_count += 1
                    continue
                
                # Get the face region
                (x, y, w, h) = faces[0]
                face_img = gray[y:y+h, x:x+w]
                
                # Apply preprocessing
                face_img = cv2.resize(face_img, (100, 100))
                face_img = cv2.equalizeHist(face_img)
                
                # Save processed image
                new_img_path = os.path.join(person_dir, f"{processed_count}.jpg")
                cv2.imwrite(new_img_path, face_img)
                
                # Add to training data
                all_face_data.append(face_img)
                all_face_labels.append(person_id)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                skipped_count += 1
        
        print(f"Processed {processed_count} images for {name}")
    
    print(f"\nTotal images processed: {len(all_face_data)}")
    print(f"Total images skipped: {skipped_count}")
    
    # Convert lists to numpy arrays for training
    X = np.array(all_face_data)
    y = np.array(all_face_labels)
    
    return {
        "X": X,
        "y": y,
        "label_names": label_names,
        "processed_dir": data_dir
    }

# STEP 4: Train model with improved parameters
def train_improved_model(data):
    """Train face recognition model with improved parameters"""
    print("\n=== Training Improved Face Recognition Model ===")
    
    if data is None or len(data["X"]) == 0:
        print("No valid data available for training.")
        return False
    
    # Make output directory
    os.makedirs("model", exist_ok=True)
    
    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], test_size=0.2, random_state=42, stratify=data["y"]
    )
    
    print(f"Training data size: {len(X_train)}")
    print(f"Validation data size: {len(X_test)}")
    
    # Create and train face recognizer
    print("Training model...")
    
    # Create Eigenfaces and Fisherfaces recognizers (LBPH removed)
    recognizers = {
        "Eigenfaces": cv2.face.EigenFaceRecognizer_create(
            num_components=80,
            threshold=5000.0
        ),
        "Fisherfaces": cv2.face.FisherFaceRecognizer_create(
            num_components=0,
            threshold=3000.0
        )
    }
    
    # Train each recognizer
    for name, recognizer in recognizers.items():
        print(f"\nTraining {name} recognizer...")
        start_time = cv2.getTickCount()
        
        try:
            recognizer.train(X_train, y_train)
            end_time = cv2.getTickCount()
            train_time = (end_time - start_time) / cv2.getTickFrequency()
            print(f"Training time: {train_time:.2f} seconds")
            
            # Save the trained model
            recognizer.save(f"model/face_model_{name.lower()}.yml")
            print(f"Model saved to model/face_model_{name.lower()}.yml")
            
            # Test on validation set
            correct = 0
            total = 0
            confidences = []
            
            print("Evaluating on validation set...")
            for i in tqdm(range(len(X_test))):
                face = X_test[i]
                actual_label = y_test[i]
                
                # Predict
                predicted_label, confidence = recognizer.predict(face)
                confidences.append(confidence)
                
                if predicted_label == actual_label:
                    correct += 1
                total += 1
            
            accuracy = (correct / total) * 100
            print(f"Validation accuracy: {accuracy:.2f}%")
            print(f"Average confidence: {np.mean(confidences):.2f}")
            print(f"Confidence range: {np.min(confidences):.2f} - {np.max(confidences):.2f}")
            
        except Exception as e:
            print(f"Error training {name} recognizer: {str(e)}")
    
    # Save label mapping
    print("\nSaving label mapping...")
    with open("model/face_labels.pickle", "wb") as f:
        pickle.dump(data["label_names"], f)
    
    print("Label mapping saved to model/face_labels.pickle")
    print("Training complete!")
    
    return True

# STEP 5: Visualize samples and results
def visualize_data(data):
    """Visualize sample processed faces"""
    if data is None or len(data["X"]) == 0:
        print("No data available for visualization.")
        return
    
    print("\n=== Visualizing Sample Faces ===")
    
    # Get a random sample of faces for each person
    unique_labels = np.unique(data["y"])
    plt.figure(figsize=(12, 4 * len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        # Get faces for this person
        indices = np.where(data["y"] == label)[0]
        
        # Sample up to 5 images
        sample_size = min(5, len(indices))
        samples = np.random.choice(indices, size=sample_size, replace=False)
        
        # Display images
        for j, idx in enumerate(samples):
            plt.subplot(len(unique_labels), 5, i * 5 + j + 1)
            plt.imshow(data["X"][idx], cmap='gray')
            plt.title(f"{data['label_names'][label]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("processed_face_samples.png")
    print("Sample visualization saved to processed_face_samples.png")

# STEP 6: Run model diagnostic tools
def run_diagnostics():
    """Run diagnostic tools to help with model troubleshooting"""
    print("\n=== Running Model Diagnostics ===")
    
    # Check model files exist
    model_files = [
        "model/face_model_eigenfaces.yml", 
        "model/face_model_fisherfaces.yml",
        "model/face_labels.pickle"
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: Some model files are missing:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("All model files exist.")
    
    # Load label names
    try:
        with open("model/face_labels.pickle", "rb") as f:
            label_names = pickle.load(f)
        print("Label mapping loaded successfully.")
        print("Registered persons in the model:")
        for person_id, name in label_names.items():
            print(f"  ID {person_id}: {name}")
    except Exception as e:
        print(f"Error loading label mapping: {str(e)}")
    
    # Run a simple test with a test image if available
    processed_dir = "processed_data"
    if os.path.exists(processed_dir):
        # Try to find a test image
        subdirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        if subdirs:
            test_dir = os.path.join(processed_dir, subdirs[0])
            test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if test_images:
                test_img_path = os.path.join(test_dir, test_images[0])
                print(f"\nRunning quick test with image: {test_img_path}")
                
                # Load image
                test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
                
                # Test with both models
                for model_type in ["eigenfaces", "fisherfaces"]:
                    model_path = f"model/face_model_{model_type}.yml"
                    if not os.path.exists(model_path):
                        continue
                    
                    try:
                        print(f"\nTesting {model_type.upper()} model:")
                        recognizer = None
                        
                        if model_type == "eigenfaces":
                            recognizer = cv2.face.EigenFaceRecognizer_create()
                        elif model_type == "fisherfaces": 
                            recognizer = cv2.face.FisherFaceRecognizer_create()
                        
                        recognizer.read(model_path)
                        
                        # Predict
                        label, confidence = recognizer.predict(test_img)
                        name = label_names.get(label, "Unknown")
                        print(f"Prediction: {name} (ID: {label})")
                        print(f"Confidence: {confidence:.2f}")
                        
                        # Suggest threshold
                        # Higher is better for Eigenfaces/Fisherfaces
                        suggested_threshold = confidence * 0.8
                        print(f"Suggested threshold for {model_type}: {suggested_threshold:.1f}")
                        
                    except Exception as e:
                        print(f"Error testing {model_type} model: {str(e)}")

# STEP 7: Show recommendations
def show_recommendations():
    """Show recommendations to improve face recognition"""
    print("\n=== Recommendations for Improved Recognition ===")
    
    recommendations = [
        "1. For Eigenfaces, try threshold values around 4000-5000. Higher values are stricter.",
        "2. For Fisherfaces, try threshold values around 3000-4000. Higher values are stricter.",
        "3. Fisherfaces generally performs better with variations in lighting conditions.",
        "4. Eigenfaces works better when lighting conditions are consistent.",
        "5. Make sure to capture face images in different lighting conditions.",
        "6. Keep the face centered and with neutral expressions for training images.",
        "7. Try to use at least 20-30 images per person for good recognition.",
        "8. Ensure good lighting during both training and recognition phases.",
        "9. Keep the same distance from camera during training and recognition."
    ]
    
    for rec in recommendations:
        print(rec)

# Main function
def main():
    print("=" * 60)
    print("Face Recognition Data Labeler & Model Retrainer")
    print("=" * 60)
    
    # Check if database is valid
    if not check_database():
        return
    
    # Verify image folders
    valid_persons = verify_image_folders()
    if not valid_persons:
        print("No valid persons with images found. Please add persons and collect face data.")
        return
    
    # Ask user if they want to continue
    response = input("\nReady to reorganize data and retrain model? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Reorganize data
    data = reorganize_data(valid_persons)
    
    # Visualize data samples
    visualize_data(data)
    
    # Train improved model
    if train_improved_model(data):
        # Run diagnostics
        run_diagnostics()
        
        # Show recommendations
        show_recommendations()
        
        print("\n=== Process Complete ===")
        print("You can now run the face recognition script to test the improved model.")
        print("Remember to adjust the model type and threshold based on the recommendations.")

if __name__ == "__main__":
    main()