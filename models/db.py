"""
models/db.py — SQLAlchemy models for College Management System
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin

db = SQLAlchemy()
bcrypt = Bcrypt()


# ── Department ─────────────────────────────────────────────────────────────────
class Department(db.Model):
    __tablename__ = "departments"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

    students = db.relationship("Student", backref="department", lazy=True)
    staff_members = db.relationship("Staff", backref="department", lazy=True)
    hods = db.relationship("HOD", backref="department", lazy=True)
    camera_alerts = db.relationship("CameraAlert", backref="department", lazy=True)


# ── Subject ────────────────────────────────────────────────────────────────────
class Subject(db.Model):
    __tablename__ = "subjects"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

    marks = db.relationship("Mark", backref="subject", lazy=True)


# ── Student ────────────────────────────────────────────────────────────────────
class Student(db.Model, UserMixin):
    __tablename__ = "students"
    id = db.Column(db.Integer, primary_key=True)
    si_no = db.Column(db.Integer, unique=True, nullable=False)
    roll_no = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    dept_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=False)
    section = db.Column(db.Enum("A", "B"), nullable=False)
    phone = db.Column(db.String(15))
    email = db.Column(db.String(100))
    address = db.Column(db.Text)
    photo = db.Column(db.String(255), default="default_student.png")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    marks = db.relationship("Mark", backref="student", lazy=True, cascade="all, delete-orphan")
    attendance = db.relationship("Attendance", backref="student", lazy=True, cascade="all, delete-orphan")

    def get_id(self):
        return f"student_{self.id}"

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


# ── Staff ──────────────────────────────────────────────────────────────────────
class Staff(db.Model, UserMixin):
    __tablename__ = "staff"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    dept_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=False)
    qualification = db.Column(db.Enum("BE", "ME", "PhD"), nullable=False)
    phone = db.Column(db.String(15))
    email = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    attendance_marked = db.relationship("Attendance", backref="marked_by_staff", lazy=True)
    announcements = db.relationship("Announcement", backref="staff_author",
                                    primaryjoin="and_(Announcement.posted_by_role=='staff', "
                                                "Announcement.posted_by_id==Staff.id)",
                                    foreign_keys="[Announcement.posted_by_id]", lazy=True)

    def get_id(self):
        return f"staff_{self.id}"

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


# ── HOD ───────────────────────────────────────────────────────────────────────
class HOD(db.Model, UserMixin):
    __tablename__ = "hods"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    dept_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=True)
    phone = db.Column(db.String(15))
    email = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_id(self):
        return f"hod_{self.id}"

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


# ── Principal ──────────────────────────────────────────────────────────────────
class Principal(db.Model, UserMixin):
    __tablename__ = "principals"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15))
    email = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_id(self):
        return f"principal_{self.id}"

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


# ── Mark ───────────────────────────────────────────────────────────────────────
class Mark(db.Model):
    __tablename__ = "marks"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey("subjects.id"), nullable=False)
    internal1 = db.Column(db.Integer, default=0)
    internal2 = db.Column(db.Integer, default=0)
    internal3 = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint("student_id", "subject_id", name="uq_student_subject"),)

    def pf(self, internal_num):
        """Return 'P' or 'F' for a given internal number (1, 2, or 3)."""
        val = getattr(self, f"internal{internal_num}", 0) or 0
        return "P" if val >= 35 else "F"


# ── Attendance ─────────────────────────────────────────────────────────────────
class Attendance(db.Model):
    __tablename__ = "attendance"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.Enum("P", "A"), nullable=False, default="P")
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint("student_id", "date", name="uq_student_date"),)


# ── Announcement ───────────────────────────────────────────────────────────────
class Announcement(db.Model):
    __tablename__ = "announcements"
    id = db.Column(db.Integer, primary_key=True)
    posted_by_role = db.Column(db.Enum("staff", "hod", "principal"), nullable=False)
    posted_by_id = db.Column(db.Integer, nullable=False)
    posted_by_name = db.Column(db.String(100))
    dept_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=True)  # NULL = all depts
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    department = db.relationship("Department", backref="announcements", foreign_keys=[dept_id])


# ── CameraAlert ────────────────────────────────────────────────────────────────
class CameraAlert(db.Model):
    __tablename__ = "camera_alerts"
    id = db.Column(db.Integer, primary_key=True)
    dept_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)   # fire, fighting, sleeping, eating, playing
    snapshot_path = db.Column(db.String(255))
    camera_index = db.Column(db.Integer, default=0)
    acknowledged = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
