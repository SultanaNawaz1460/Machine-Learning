"""
Face Recognition System - Database Setup
This script creates a database to store person information.
"""
import os
import sqlite3
import pandas as pd

def create_database():
    # Create database connection
    conn = sqlite3.connect('face_recognition_db.sqlite')
    cursor = conn.cursor()
    
    # Create person table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        department TEXT,
        semester TEXT,
        registration_number TEXT NOT NULL,
        image_folder TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database created successfully!")
    return True

def add_person(name, age, gender, department, semester, registration_number, image_folder):
    conn = sqlite3.connect('face_recognition_db.sqlite')
    cursor = conn.cursor()
    
    # Check if person already exists
    cursor.execute("SELECT * FROM persons WHERE name = ?", (name,))
    if cursor.fetchone():
        print(f"Person '{name}' already exists in the database!")
        conn.close()
        return False
    
    # Add new person
    cursor.execute('''
    INSERT INTO persons (name, age, gender, department, semester, registration_number, image_folder)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, age, gender, department, semester, registration_number, image_folder))
    
    conn.commit()
    conn.close()
    
    print(f"Person '{name}' added successfully!")
    return True

def list_all_persons():
    conn = sqlite3.connect('face_recognition_db.sqlite')
    
    # Read all persons into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM persons", conn)
    conn.close()
    
    if df.empty:
        print("No persons found in the database!")
    else:
        print("\n=== Registered Persons ===")
        print(df.to_string(index=False))
    
    return df

def main():
    # Create database if it doesn't exist
    create_database()
    
    while True:
        print("\n=== Person Database Management ===")
        print("1. Add new person")
        print("2. List all persons")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            print("\n=== Add New Person ===")
            name = input("Name: ")
            age = int(input("Age: "))
            gender = input("Gender (M/F): ")
            department = input("Department: ")
            semester = input("Semester: ")
            registration_number = input("Registration Number: ")
            
            # Create folder for person's images if it doesn't exist
            image_folder = os.path.join("dataset", name.replace(" ", "_").lower())
            os.makedirs(image_folder, exist_ok=True)
            
            add_person(name, age, gender, department, semester, registration_number, image_folder)
            
        elif choice == '2':
            list_all_persons()
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
