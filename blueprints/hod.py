"""
blueprints/hod.py — HOD dashboard blueprint
"""
from datetime import date
from flask import Blueprint, render_template, session, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from flask_mail import Message
from models.db import db, Student, Staff, Mark, Subject, Attendance, Announcement, Department, CameraAlert, HOD

hod_bp = Blueprint("hod", __name__)


def hod_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "hod":
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


@hod_bp.route("/dashboard")
@login_required
@hod_required
def dashboard():
    hod  = current_user
    dept = Department.query.get(hod.dept_id) if hod.dept_id else None

    students      = Student.query.filter_by(dept_id=hod.dept_id).order_by(Student.section, Student.roll_no).all() if hod.dept_id else []
    staff_members = Staff.query.filter_by(dept_id=hod.dept_id).order_by(Staff.qualification).all() if hod.dept_id else []

    unacked_alerts = (CameraAlert.query
                      .filter_by(dept_id=hod.dept_id, acknowledged=False)
                      .order_by(CameraAlert.created_at.desc()).all()) if hod.dept_id else []

    qual_filter = request.args.get("qual", "all")
    if qual_filter != "all" and hod.dept_id:
        staff_members = Staff.query.filter_by(dept_id=hod.dept_id, qualification=qual_filter).all()

    return render_template("hod/dashboard.html",
                           hod=hod, dept=dept,
                           students=students,
                           staff_members=staff_members,
                           unacked_alerts=unacked_alerts,
                           alert_count=len(unacked_alerts),
                           qual_filter=qual_filter,
                           today=date.today().isoformat())


@hod_bp.route("/students")
@login_required
@hod_required
def students():
    hod     = current_user
    dept    = Department.query.get(hod.dept_id)
    section = request.args.get("section", "all")
    query   = Student.query.filter_by(dept_id=hod.dept_id)
    if section != "all":
        query = query.filter_by(section=section)
    students = query.order_by(Student.section, Student.roll_no).all()
    return render_template("hod/students.html", hod=hod, dept=dept, students=students, section=section)


@hod_bp.route("/students/edit/<int:student_id>", methods=["GET", "POST"])
@login_required
@hod_required
def edit_student(student_id):
    hod = current_user
    student = Student.query.get_or_404(student_id)
    
    if student.dept_id != hod.dept_id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for("hod.students"))
        
    if request.method == "POST":
        student.name = request.form.get("name", student.name).strip()
        student.roll_no = request.form.get("roll_no", student.roll_no).strip()
        student.section = request.form.get("section", student.section).strip()
        student.phone = request.form.get("phone", student.phone).strip()
        student.email = request.form.get("email", student.email).strip()
        student.address = request.form.get("address", student.address).strip()
        student.photo = request.form.get("photo", student.photo).strip()
        
        db.session.commit()
        flash("Student updated successfully.", "success")
        return redirect(url_for("hod.students"))
        
    dept = Department.query.get(hod.dept_id)
    return render_template("hod/edit_student.html", hod=hod, dept=dept, student=student)


@hod_bp.route("/staff")
@login_required
@hod_required
def staff_list():
    hod   = current_user
    dept  = Department.query.get(hod.dept_id)
    qual  = request.args.get("qual", "all")
    query = Staff.query.filter_by(dept_id=hod.dept_id)
    if qual != "all":
        query = query.filter_by(qualification=qual)
    staff_members = query.order_by(Staff.qualification, Staff.name).all()
    counts = {
        "BE":  Staff.query.filter_by(dept_id=hod.dept_id, qualification="BE").count(),
        "ME":  Staff.query.filter_by(dept_id=hod.dept_id, qualification="ME").count(),
        "PhD": Staff.query.filter_by(dept_id=hod.dept_id, qualification="PhD").count(),
    }
    return render_template("hod/staff.html", hod=hod, dept=dept,
                           staff_members=staff_members, qual=qual, counts=counts)


@hod_bp.route("/marks")
@login_required
@hod_required
def marks():
    hod      = current_user
    dept     = Department.query.get(hod.dept_id)
    section  = request.args.get("section", "A")
    students = Student.query.filter_by(dept_id=hod.dept_id, section=section).order_by(Student.roll_no).all()
    subjects = Subject.query.order_by(Subject.id).all()
    marks_dict = {}
    for stu in students:
        marks_dict[stu.id] = {mk.subject_id: mk for mk in Mark.query.filter_by(student_id=stu.id).all()}
    return render_template("hod/marks.html", hod=hod, dept=dept,
                           students=students, subjects=subjects,
                           marks_dict=marks_dict, section=section)


@hod_bp.route("/attendance")
@login_required
@hod_required
def attendance():
    hod      = current_user
    dept     = Department.query.get(hod.dept_id)
    sel_date = request.args.get("date", date.today().isoformat())
    section  = request.args.get("section", "A")
    students = Student.query.filter_by(dept_id=hod.dept_id, section=section).order_by(Student.roll_no).all()
    existing = {att.student_id: att.status
                for att in Attendance.query.filter_by(date=sel_date).all()}
    return render_template("hod/attendance.html", hod=hod, dept=dept,
                           students=students, sel_date=sel_date,
                           section=section, existing=existing)


@hod_bp.route("/announce", methods=["GET", "POST"])
@login_required
@hod_required
def announce():
    hod  = current_user
    dept = Department.query.get(hod.dept_id)
    if request.method == "POST":
        title   = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title and content:
            ann = Announcement(posted_by_role="hod", posted_by_id=hod.id,
                               posted_by_name=hod.name, dept_id=hod.dept_id,
                               title=title, content=content)
            db.session.add(ann)
            db.session.commit()
            flash("Announcement posted!", "success")
        else:
            flash("Title and content required.", "danger")
        return redirect(url_for("hod.announce"))
    announcements = (Announcement.query
                     .filter_by(posted_by_role="hod", dept_id=hod.dept_id)
                     .order_by(Announcement.created_at.desc()).limit(20).all())
    return render_template("hod/announce.html", hod=hod, dept=dept, announcements=announcements)


@hod_bp.route("/alerts")
@login_required
@hod_required
def alerts():
    hod    = current_user
    dept   = Department.query.get(hod.dept_id)
    alerts = (CameraAlert.query.filter_by(dept_id=hod.dept_id)
              .order_by(CameraAlert.created_at.desc()).limit(50).all())
    return render_template("hod/alerts.html", hod=hod, dept=dept, alerts=alerts)


@hod_bp.route("/alerts/ack/<int:alert_id>", methods=["POST"])
@login_required
@hod_required
def ack_alert(alert_id):
    alert = CameraAlert.query.get_or_404(alert_id)
    if alert.dept_id == current_user.dept_id:
        alert.acknowledged = True
        db.session.commit()
    return jsonify({"status": "ok"})


@hod_bp.route("/camera")
@login_required
@hod_required
def camera():
    hod   = current_user
    dept  = Department.query.get(hod.dept_id)
    alerts = (CameraAlert.query.filter_by(dept_id=hod.dept_id)
              .order_by(CameraAlert.created_at.desc()).limit(20).all())
    return render_template("hod/camera.html", hod=hod, dept=dept, alerts=alerts)


@hod_bp.route("/api/notifications/count")
@login_required
@hod_required
def notif_count():
    hod   = current_user
    count = CameraAlert.query.filter_by(dept_id=hod.dept_id, acknowledged=False).count()
    return jsonify({"count": count})
