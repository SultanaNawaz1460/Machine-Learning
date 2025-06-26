"""
Face Recognition System - Data Analysis and Processing
This script analyzes face images and processes them for improved recognition.
"""
import os
import cv2
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import shutil

def analyze_database():
    """Analyze the face recognition database"""
    print("\n=== Analyzing Database ===")
    
    if not os.path.exists('face_recognition_db.sqlite'):
        print("Database not found! Please run the database setup script first.")
        return None
    
    conn = sqlite3.connect('face_recognition_db.sqlite')
    cursor = conn.cursor()
    
    # Check if persons table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='persons'")
    if not cursor.fetchone():
        print("Database structure is invalid - missing persons table!")
        conn.close()
        return None
    
    # Get persons count
    cursor.execute("SELECT COUNT(*) FROM persons")
    count = cursor.fetchone()[0]
    print(f"Found {count} persons in database.")
    
    # Get all persons
    persons_df = pd.read_sql_query("SELECT * FROM persons", conn)
    conn.close()
    
    if persons_df.empty:
        print("No persons found in database!")
        return None
    
    print("\nRegistered persons:")
    print(persons_df[['id', 'name', 'age', 'gender', 'department', 'image_folder']])
    
    return persons_df

def analyze_face_data(persons_df):
    """Analyze face data for each person"""
    print("\n=== Analyzing Face Data ===")
    
    if persons_df is None or persons_df.empty:
        return None
    
    results = []
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Failed to load face detector cascade.")
        return None
    
    for _, person in persons_df.iterrows():
        person_id = person['id']
        name = person['name']
        folder = person['image_folder']
        
        print(f"\nAnalyzing images for {name} (ID: {person_id})...")
        
        if not os.path.exists(folder):
            print(f"Warning: Folder does not exist: {folder}")
            results.append({
                'person_id': person_id,
                'name': name,
                'total_images': 0,
                'valid_faces': 0,
                'multiple_faces': 0,
                'no_faces': 0,
                'avg_face_size': 0,
                'avg_brightness': 0,
                'avg_contrast': 0,
                'folder_exists': False
            })
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {folder}")
            results.append({
                'person_id': person_id,
                'name': name,
                'total_images': 0,
                'valid_faces': 0,
                'multiple_faces': 0,
                'no_faces': 0,
                'avg_face_size': 0,
                'avg_brightness': 0,
                'avg_contrast': 0,
                'folder_exists': True
            })
            continue
        
        print(f"Found {len(image_files)} images")
        
        # Statistics
        valid_faces = 0
        multiple_faces = 0
        no_faces = 0
        face_sizes = []
        brightness_values = []
        contrast_values = []
        
        # Process images
        for img_file in tqdm(image_files):
            img_path = os.path.join(folder, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Could not read {img_file}")
                    continue
                
                # Calculate brightness and contrast
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                contrast = np.std(gray)
                
                brightness_values.append(brightness)
                contrast_values.append(contrast)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) == 0:
                    no_faces += 1
                elif len(faces) == 1:
                    valid_faces += 1
                    (x, y, w, h) = faces[0]
                    face_sizes.append(w * h)
                else:
                    multiple_faces += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
        
        # Calculate averages
        avg_face_size = np.mean(face_sizes) if face_sizes else 0
        avg_brightness = np.mean(brightness_values) if brightness_values else 0
        avg_contrast = np.mean(contrast_values) if contrast_values else 0
        
        results.append({
            'person_id': person_id,
            'name': name,
            'total_images': len(image_files),
            'valid_faces': valid_faces,
            'multiple_faces': multiple_faces,
            'no_faces': no_faces,
            'avg_face_size': avg_face_size,
            'avg_brightness': avg_brightness,
            'avg_contrast': avg_contrast,
            'folder_exists': True
        })
        
        print(f"Results for {name}:")
        print(f"  Valid faces: {valid_faces}/{len(image_files)} ({valid_faces/len(image_files)*100:.1f}%)")
        print(f"  Multiple faces: {multiple_faces}/{len(image_files)} ({multiple_faces/len(image_files)*100:.1f}%)")
        print(f"  No faces: {no_faces}/{len(image_files)} ({no_faces/len(image_files)*100:.1f}%)")
        
        if face_sizes:
            print(f"  Average face size: {avg_face_size:.1f} pixels²")
        if brightness_values:
            print(f"  Average brightness: {avg_brightness:.1f}/255")
        if contrast_values:
            print(f"  Average contrast: {avg_contrast:.1f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def visualize_sample_faces(persons_df, results_df, save_path="sample_faces.png"):
    """Visualize sample faces from each person's dataset"""
    print("\n=== Visualizing Sample Faces ===")
    
    if persons_df is None or persons_df.empty:
        return
    
    # Create a figure
    n_persons = len(persons_df)
    plt.figure(figsize=(15, 3 * n_persons))
    
    # For each person, show a few sample faces
    for i, (_, person) in enumerate(persons_df.iterrows()):
        name = person['name']
        folder = person['image_folder']
        
        if not os.path.exists(folder):
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to find up to 5 images with good faces
        sample_images = []
        face_images = []
        
        # Shuffle image files to get different samples each time
        np.random.shuffle(image_files)
        
        for img_file in image_files:
            if len(sample_images) >= 5:
                break
                
            img_path = os.path.join(folder, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) == 1:
                    sample_images.append(img)
                    
                    # Extract face
                    (x, y, w, h) = faces[0]
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (100, 100))
                    face = cv2.equalizeHist(face)
                    face_images.append(face)
                
            except Exception:
                continue
        
        # Plot sample images
        for j, (sample, face) in enumerate(zip(sample_images, face_images)):
            # Original image
            plt.subplot(n_persons, 10, i*10 + j*2 + 1)
            plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
            plt.title(f"{name} (original)" if j == 0 else "")
            plt.axis('off')
            
            # Processed face
            plt.subplot(n_persons, 10, i*10 + j*2 + 2)
            plt.imshow(face, cmap='gray')
            plt.title(f"{name} (processed)" if j == 0 else "")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Sample faces saved to {save_path}")

    plt.figure(figsize=(12, 6))
    
    # Create bar chart for face quality
    plt.subplot(1, 2, 1)
    valid_percentages = []
    names = []
    
    for _, row in results_df.iterrows():
        if row['total_images'] > 0:
            valid_percentages.append(row['valid_faces'] / row['total_images'] * 100)
            names.append(row['name'])
    
    plt.bar(names, valid_percentages, color='green')
    plt.title('Percentage of Valid Face Images')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    # Create bar chart for image counts
    plt.subplot(1, 2, 2)
    valid_faces = results_df['valid_faces'].tolist()
    total_images = results_df['total_images'].tolist()
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, total_images, width, label='Total Images')
    plt.bar(x + width/2, valid_faces, width, label='Valid Faces')
    
    plt.title('Image Counts by Person')
    plt.ylabel('Count')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("face_data_statistics.png")
    print("Face data statistics saved to face_data_statistics.png")

def preprocess_images(persons_df):
    """Preprocess all face images to improve recognition"""
    print("\n=== Preprocessing Face Images ===")
    
    if persons_df is None or persons_df.empty:
        return
    
    # Create preprocessed data directory
    preprocessed_dir = "preprocessed_data"
    if os.path.exists(preprocessed_dir):
        response = input(f"Directory {preprocessed_dir} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Preprocessing cancelled.")
            return
        shutil.rmtree(preprocessed_dir)
    
    os.makedirs(preprocessed_dir)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Failed to load face detector cascade.")
        return
    
    total_processed = 0
    
    for _, person in persons_df.iterrows():
        person_id = person['id']
        name = person['name']
        folder = person['image_folder']
        
        print(f"\nPreprocessing images for {name} (ID: {person_id})...")
        
        if not os.path.exists(folder):
            print(f"Warning: Folder does not exist: {folder}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {folder}")
            continue
        
        # Create person directory in preprocessed data
        person_dir = os.path.join(preprocessed_dir, f"{person_id}_{name.replace(' ', '_')}")
        os.makedirs(person_dir)
        
        processed_count = 0
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(folder, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization to improve contrast
                gray = cv2.equalizeHist(gray)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) != 1:
                    continue
                
                # Get face region
                (x, y, w, h) = faces[0]
                
                # Extract face
                face = gray[y:y+h, x:x+w]
                
                # Resize to standard size
                face = cv2.resize(face, (100, 100))
                
                # Apply additional preprocessing
                face = cv2.equalizeHist(face)  # Enhance contrast
                
                # Save preprocessed face
                output_path = os.path.join(person_dir, f"face_{processed_count}.jpg")
                cv2.imwrite(output_path, face)
                
                processed_count += 1
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
        
        print(f"Processed {processed_count} images for {name}")
    
    print(f"\nTotal preprocessed images: {total_processed}")
    print(f"Preprocessed data saved to {preprocessed_dir}")
    
    return preprocessed_dir

def main():
    print("=" * 60)
    print("Face Data Analysis and Processing Tool")
    print("=" * 60)
    
    # Analyze database
    persons_df = analyze_database()
    if persons_df is None:
        return
    
    # Analyze face data
    results_df = analyze_face_data(persons_df)
    if results_df is None:
        return
    
    # Visualize sample faces
    visualize_sample_faces(persons_df, results_df)
    
    # Ask if user wants to preprocess images
    response = input("\nDo you want to preprocess all face images? (y/n): ")
    if response.lower() == 'y':
        preprocess_images(persons_df)
    
    print("\nAnalysis complete. Please check the generated images for visualization.")

if __name__ == "__main__":
    main()