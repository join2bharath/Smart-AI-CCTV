"""
blueprints/staff.py — Staff dashboard blueprint
"""
from datetime import date, datetime
from flask import Blueprint, render_template, session, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from models.db import db, Student, Mark, Subject, Attendance, Announcement, Department, CameraAlert

staff_bp = Blueprint("staff", __name__)


def staff_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "staff":
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


@staff_bp.route("/dashboard")
@login_required
@staff_required
def dashboard():
    staff = current_user
    dept = Department.query.get(staff.dept_id)

    students = (Student.query
                .filter_by(dept_id=staff.dept_id)
                .order_by(Student.section, Student.roll_no)
                .all())

    recent_alerts = (CameraAlert.query
                     .filter_by(dept_id=staff.dept_id, acknowledged=False)
                     .order_by(CameraAlert.created_at.desc())
                     .limit(5).all())

    return render_template("staff/dashboard.html",
                           staff=staff, dept=dept,
                           students=students,
                           recent_alerts=recent_alerts,
                           today=date.today().isoformat())


@staff_bp.route("/attendance", methods=["GET", "POST"])
@login_required
@staff_required
def attendance():
    staff = current_user
    dept  = Department.query.get(staff.dept_id)
    sel_date = request.args.get("date", date.today().isoformat())
    section  = request.args.get("section", "A")

    students = (Student.query
                .filter_by(dept_id=staff.dept_id, section=section)
                .order_by(Student.roll_no).all())

    # existing attendance for this date
    existing = {}
    for att in Attendance.query.filter_by(date=sel_date).all():
        existing[att.student_id] = att.status

    if request.method == "POST":
        sel_date = request.form.get("date", date.today().isoformat())
        section  = request.form.get("section", "A")
        students = (Student.query
                    .filter_by(dept_id=staff.dept_id, section=section)
                    .order_by(Student.roll_no).all())
        for stu in students:
            status = request.form.get(f"status_{stu.id}", "A")
            att = Attendance.query.filter_by(student_id=stu.id, date=sel_date).first()
            if att:
                att.status   = status
                att.staff_id = staff.id
            else:
                att = Attendance(student_id=stu.id, date=sel_date,
                                 status=status, staff_id=staff.id)
                db.session.add(att)
        db.session.commit()
        flash("Attendance saved successfully!", "success")
        return redirect(url_for("staff.attendance", date=sel_date, section=section))

    return render_template("staff/attendance.html",
                           staff=staff, dept=dept, students=students,
                           sel_date=sel_date, section=section,
                           existing=existing)


@staff_bp.route("/marks", methods=["GET", "POST"])
@login_required
@staff_required
def marks():
    staff    = current_user
    dept     = Department.query.get(staff.dept_id)
    section  = request.args.get("section", "A")
    students = (Student.query
                .filter_by(dept_id=staff.dept_id, section=section)
                .order_by(Student.roll_no).all())
    subjects = Subject.query.order_by(Subject.id).all()

    if request.method == "POST":
        for stu in students:
            for subj in subjects:
                for i in (1, 2, 3):
                    val = request.form.get(f"mark_{stu.id}_{subj.id}_{i}", "0")
                    val = max(0, min(100, int(val) if val.isdigit() else 0))
                    mk  = Mark.query.filter_by(student_id=stu.id, subject_id=subj.id).first()
                    if mk:
                        setattr(mk, f"internal{i}", val)
                    else:
                        mk = Mark(student_id=stu.id, subject_id=subj.id)
                        setattr(mk, f"internal{i}", val)
                        db.session.add(mk)
        db.session.commit()
        flash("Marks updated successfully!", "success")
        return redirect(url_for("staff.marks", section=section))

    # Build marks dict {student_id: {subject_id: Mark}}
    marks_dict = {}
    for stu in students:
        marks_dict[stu.id] = {}
        for mk in Mark.query.filter_by(student_id=stu.id).all():
            marks_dict[stu.id][mk.subject_id] = mk

    return render_template("staff/marks.html",
                           staff=staff, dept=dept,
                           students=students, subjects=subjects,
                           marks_dict=marks_dict, section=section)


@staff_bp.route("/announce", methods=["GET", "POST"])
@login_required
@staff_required
def announce():
    staff = current_user
    dept  = Department.query.get(staff.dept_id)

    if request.method == "POST":
        title   = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title and content:
            ann = Announcement(
                posted_by_role="staff",
                posted_by_id=staff.id,
                posted_by_name=staff.name,
                dept_id=staff.dept_id,
                title=title,
                content=content,
            )
            db.session.add(ann)
            db.session.commit()
            flash("Announcement posted!", "success")
        else:
            flash("Title and content are required.", "danger")
        return redirect(url_for("staff.announce"))

    announcements = (Announcement.query
                     .filter_by(posted_by_role="staff", dept_id=staff.dept_id)
                     .order_by(Announcement.created_at.desc()).limit(20).all())

    return render_template("staff/announce.html",
                           staff=staff, dept=dept,
                           announcements=announcements)


@staff_bp.route("/camera")
@login_required
@staff_required
def camera():
    staff = current_user
    dept  = Department.query.get(staff.dept_id)
    alerts = (CameraAlert.query
              .filter_by(dept_id=staff.dept_id)
              .order_by(CameraAlert.created_at.desc())
              .limit(20).all())
    return render_template("staff/camera.html", staff=staff, dept=dept, alerts=alerts)


@staff_bp.route("/api/notifications/count")
@login_required
@staff_required
def notif_count():
    staff = current_user
    count = CameraAlert.query.filter_by(dept_id=staff.dept_id, acknowledged=False).count()
    return jsonify({"count": count})


@staff_bp.route("/alerts/ack/<int:alert_id>", methods=["POST"])
@login_required
@staff_required
def ack_alert(alert_id):
    """Acknowledge a camera alert (staff can only ack their own dept)."""
    alert = CameraAlert.query.get_or_404(alert_id)
    if alert.dept_id == current_user.dept_id:
        alert.acknowledged = True
        db.session.commit()
    return jsonify({"status": "ok"})


@staff_bp.route("/api/camera-stats")
@login_required
@staff_required
def camera_stats():
    """Return live camera stats for the staff's department."""
    staff  = current_user
    latest = (CameraAlert.query
              .filter_by(dept_id=staff.dept_id)
              .order_by(CameraAlert.created_at.desc())
              .first())
    sleeping = (CameraAlert.query
                .filter_by(dept_id=staff.dept_id, alert_type="sleeping", acknowledged=False)
                .count())
    fire     = (CameraAlert.query
                .filter_by(dept_id=staff.dept_id, alert_type="fire", acknowledged=False)
                .count())
    fighting = (CameraAlert.query
                .filter_by(dept_id=staff.dept_id, alert_type="fighting", acknowledged=False)
                .count())
    unacked  = (CameraAlert.query
                .filter_by(dept_id=staff.dept_id, acknowledged=False)
                .count())
    return jsonify({
        "unacked_alerts":  unacked,
        "sleeping_alerts": sleeping,
        "fire_alerts":     fire,
        "fighting_alerts": fighting,
        "latest_alert":    latest.alert_type if latest else None,
        "latest_time":     latest.created_at.strftime("%H:%M:%S") if latest else None,
        "alert_icons": {
            "fire": "🔥", "fighting": "👊", "sleeping": "😴",
            "eating": "🍔", "playing": "🎮", "dancing": "💃",
            "mobile_usage": "📱", "hand_raising": "✋",
            "standing": "🧍", "sitting": "🪑",
        }
    })

