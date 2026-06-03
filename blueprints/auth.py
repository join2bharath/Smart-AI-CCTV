"""
blueprints/auth.py — Login / Logout for all roles
"""
from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from models.db import db, Student, Staff, HOD, Principal

auth_bp = Blueprint("auth", __name__)


def detect_role(username: str):
    """Return (role_str, model_instance_or_None)."""
    if username.startswith("student"):
        user = Student.query.filter_by(username=username).first()
        return "student", user
    elif username.startswith("staff@"):
        user = Staff.query.filter_by(username=username).first()
        return "staff", user
    elif username.startswith("HOD@"):
        user = HOD.query.filter_by(username=username).first()
        return "hod", user
    elif username.startswith("Principal_"):
        user = Principal.query.filter_by(username=username).first()
        return "principal", user
    return None, None


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return _redirect_dashboard()

    if request.method == "POST":
        username   = request.form.get("username", "").strip()
        password   = request.form.get("password", "")
        remember   = request.form.get("remember") == "on"

        role, user = detect_role(username)
        if user and user.check_password(password):
            login_user(user, remember=remember)
            session["role"] = role
            flash(f"Welcome back, {user.name}!", "success")
            return _redirect_dashboard()
        else:
            flash("Invalid username or password. Please try again.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


def _redirect_dashboard():
    role = session.get("role")
    if role == "student":
        return redirect(url_for("student.dashboard"))
    elif role == "staff":
        return redirect(url_for("staff.dashboard"))
    elif role == "hod":
        return redirect(url_for("hod.dashboard"))
    elif role == "principal":
        return redirect(url_for("principal.dashboard"))
    return redirect(url_for("auth.login"))


@auth_bp.route("/forgot-password")
def forgot_password():
    return render_template("forgot_password.html")
