"""
blueprints/student.py — Student dashboard blueprint
"""
from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from flask_login import login_required, current_user
from models.db import db, Mark, Announcement, Subject, Department, Attendance

student_bp = Blueprint("student", __name__)


def student_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "student":
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


@student_bp.route("/dashboard")
@login_required
@student_required
def dashboard():
    student = current_user

    # Fetch marks with subjects
    marks_data = (
        db.session.query(Mark, Subject)
        .join(Subject, Mark.subject_id == Subject.id)
        .filter(Mark.student_id == student.id)
        .order_by(Subject.id)
        .all()
    )

    # Announcements: staff/hod for student's dept + all principal announcements
    staff_ann = (
        Announcement.query
        .filter_by(posted_by_role="staff", dept_id=student.dept_id)
        .order_by(Announcement.created_at.desc())
        .limit(5).all()
    )
    hod_ann = (
        Announcement.query
        .filter_by(posted_by_role="hod", dept_id=student.dept_id)
        .order_by(Announcement.created_at.desc())
        .limit(5).all()
    )
    principal_ann = (
        Announcement.query
        .filter_by(posted_by_role="principal")
        .order_by(Announcement.created_at.desc())
        .limit(5).all()
    )

    dept = Department.query.get(student.dept_id)

    return render_template(
        "student/dashboard.html",
        student=student,
        dept=dept,
        marks_data=marks_data,
        staff_ann=staff_ann,
        hod_ann=hod_ann,
        principal_ann=principal_ann,
    )


@student_bp.route("/attendance")
@login_required
@student_required
def attendance():
    student = current_user
    dept = Department.query.get(student.dept_id)
    
    # Fetch all attendance records ordered by date desc
    records = (
        Attendance.query
        .filter_by(student_id=student.id)
        .order_by(Attendance.date.desc())
        .all()
    )
    
    # Compute statistics
    total_days = len(records)
    present_days = sum(1 for r in records if r.status == "P")
    absent_days = total_days - present_days
    
    percent = 100.0
    if total_days > 0:
        percent = round((present_days / total_days) * 100, 1)
        
    return render_template(
        "student/attendance.html",
        student=student,
        dept=dept,
        records=records,
        total_days=total_days,
        present_days=present_days,
        absent_days=absent_days,
        percent=percent
    )

@student_bp.route("/profile/update", methods=["POST"])
@login_required
@student_required
def update_profile():
    student = current_user
    student.phone = request.form.get("phone", student.phone).strip()
    student.email = request.form.get("email", student.email).strip()
    student.address = request.form.get("address", student.address).strip()
    student.photo = request.form.get("photo", student.photo).strip()
    
    db.session.commit()
    flash("Profile updated successfully.", "success")
    return redirect(url_for("student.dashboard"))
