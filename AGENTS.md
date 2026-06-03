# College Management System — Agent Context

## Project Type
Full-stack web app: Python Flask + MySQL + HTML/CSS/JS + OpenCV AI camera

## Database
MySQL — host: localhost, port: 3306, user: root, password: root
DB name: college_db

## Departments
IT, CSE, AIDS, ECE, EEE, CIVIL
Each has Section A and Section B

## Users
- Students: username=student001..student1100, password=their roll number
- Staff: username=staff@01..staff@287, password=Welcome@123
- HOD: username=HOD@01..HOD@11, password=welcome@123
- Principal: username=Principal_1 or Principal_2, password=Principal@123

## Subjects
English, Physics, Chemistry, Maths, Python, Tamil

## Pass/Fail Rule
Mark >= 35 = Pass (P), else Fail (F) — computed at runtime, NOT stored

## Camera Rule
Staff and HOD can only access their own department's cameras.
Principal can access all cameras.

## Tech Constraints
- Use Flask blueprints, one per role (student, staff, hod, principal)
- All passwords stored as bcrypt hashes
- Camera alerts stored in MySQL table 'camera_alerts'
- AI camera service runs as a separate Python process (camera_service.py)
