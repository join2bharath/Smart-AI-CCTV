"""
config.py — Application configuration for College Management System
"""
import os


class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", "college-cms-secret-key-2024-change-in-prod")
    DEBUG = True

    # ── MySQL ──────────────────────────────────────────────────────────────────
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "root"
    MYSQL_DB = "college_db"

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_RECYCLE = 280
    SQLALCHEMY_POOL_TIMEOUT = 20

    # ── Flask-Mail (SMTP) ──────────────────────────────────────────────────────
    MAIL_SERVER = "smtp.gmail.com"
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "college.cms.alerts@gmail.com")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "your-app-password-here")
    MAIL_DEFAULT_SENDER = ("College CMS Alert", os.environ.get("MAIL_USERNAME", "college.cms.alerts@gmail.com"))

    # ── Subjects ───────────────────────────────────────────────────────────────
    SUBJECTS = ["English", "Physics", "Chemistry", "Maths", "Python", "Tamil"]

    # ── Departments ────────────────────────────────────────────────────────────
    DEPARTMENTS = ["IT", "CSE", "AIDS", "ECE", "EEE", "CIVIL"]
    SECTIONS = ["A", "B"]

    # ── Snapshots ──────────────────────────────────────────────────────────────
    SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

    # ── Pass/Fail threshold ────────────────────────────────────────────────────
    PASS_MARK = 35
