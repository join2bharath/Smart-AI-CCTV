# College Management System (CMS) — with AI Camera Alert System

Welcome to the College Management System (CMS) with AI-powered Camera Alert System. This application is a full-stack web application built using Python Flask, MySQL, and modern Vanilla HTML/CSS/JS, integrated with a real-time OpenCV AI camera process.

---

## 🚀 Key Features

- **Role-based Dashboards**: Customized access for **Students**, **Staff**, **Department Heads (HODs)**, and the **Principal**.
- **AI Camera Surveillance**: Runs as a separate process utilizing OpenCV and optional AI detection models to identify events such as `fire`, `fighting`, `sleeping`, `eating`, `playing`, `dancing`, or `person count` fluctuations.
- **Real-time Notifications**: Alert notifications for HODs and the Principal inside their dashboards, with snapshot captures.
- **Smart Scoping**: Staff and HODs only see camera streams and alerts for their own departments. The Principal has access to all department streams and alerts.
- **Academic Management**: Mark student attendance, manage internal test grades (computed runtime Pass/Fail rules), and post announcements.

---

## 🛠️ Technology Stack

- **Backend**: Python Flask (using Flask Blueprints for clean routing)
- **Database**: MySQL (Host: `localhost`, Port: `3306`, User: `root`, Password: `root`, DB name: `college_db`)
- **Frontend**: Responsive HTML5, Vanilla CSS3 (curated professional navy palette), and dynamic JavaScript.
- **AI Engine**: Python OpenCV, MediaPipe (Face Mesh for drowsiness detection), YOLOv8 (pose/objects detection).

---

## 👤 User Roles & Accounts

Passwords are securely stored as **bcrypt hashes** in MySQL. Default seeded credentials:

| Role | Username Pattern | Password Pattern | Permissions |
| :--- | :--- | :--- | :--- |
| **Principal** | `Principal_1`, `Principal_2` | `Principal@123` | Global monitoring, view all marks/attendance, view all alerts |
| **HOD** | `HOD@01` .. `HOD@11` | `welcome@123` | Manage dept staff/students, view dept alerts & camera streams |
| **Staff** | `staff@01` .. `staff@287` | `Welcome@123` | Mark attendance, edit internal marks, post announcements |
| **Student** | `student001` .. `student1100` | *Their respective Roll Number* (e.g. `ITA0001`) | View own attendance, internal marks, and announcements |

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have MySQL and Python 3.10+ installed and running.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize & Seed Database
Make sure your MySQL server is running on localhost:3306, then execute:
```bash
python seed.py
```
This will automatically create `college_db`, define tables, and populate:
- **6 Departments**: IT, CSE, AIDS, ECE, EEE, CIVIL (with Sections A & B)
- **6 Subjects**: English, Physics, Chemistry, Maths, Python, Tamil
- **287 Staff Members** (distributed by dept / qualifications)
- **11 HODs** (1 per dept + spares)
- **2 Principals**
- **1,100 Students** (each loaded with sample internal test scores, attendance history, and profile avatars)

### 4. Running the Web Application
Start the Flask application:
```bash
flask run
```
Access the application at `http://127.0.0.1:5000`.

### 5. Running the AI Camera Service
Run the background AI camera service:
```bash
# To run on default camera
python camera_service.py

# To run for a specific department
python camera_service.py --dept IT --cam 0
```
Alert snapshots will be automatically stored inside `static/snapshots/` and displayed instantly on the HOD/Principal dashboards!
