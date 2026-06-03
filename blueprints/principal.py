"""
blueprints/principal.py — Principal dashboard blueprint
"""
from flask import Blueprint, render_template, session, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from models.db import db, Student, Staff, HOD, Mark, Subject, Attendance, Announcement, Department, CameraAlert

principal_bp = Blueprint("principal", __name__)


def principal_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "principal":
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


@principal_bp.route("/dashboard")
@login_required
@principal_required
def dashboard():
    principal   = current_user
    departments = Department.query.all()
    dept_stats  = []
    for dept in departments:
        dept_stats.append({
            "dept":     dept,
            "students": Student.query.filter_by(dept_id=dept.id).count(),
            "staff":    Staff.query.filter_by(dept_id=dept.id).count(),
            "alerts":   CameraAlert.query.filter_by(dept_id=dept.id, acknowledged=False).count(),
        })
    total_alerts = CameraAlert.query.filter_by(acknowledged=False).count()
    recent_alerts = CameraAlert.query.order_by(CameraAlert.created_at.desc()).limit(10).all()
    recent_ann    = Announcement.query.order_by(Announcement.created_at.desc()).limit(5).all()

    return render_template("principal/dashboard.html",
                           principal=principal,
                           dept_stats=dept_stats,
                           total_alerts=total_alerts,
                           recent_alerts=recent_alerts,
                           recent_ann=recent_ann)


@principal_bp.route("/department/<int:dept_id>")
@login_required
@principal_required
def dept_view(dept_id):
    principal = current_user
    dept      = Department.query.get_or_404(dept_id)
    qual      = request.args.get("qual", "all")
    section   = request.args.get("section", "all")

    # Staff filter logic: BE → show ME+PhD, ME → show PhD, PhD → show PhD
    if qual == "BE":
        staff_list = Staff.query.filter(Staff.dept_id == dept_id,
                                        Staff.qualification.in_(["ME", "PhD"])).all()
    elif qual == "ME":
        staff_list = Staff.query.filter(Staff.dept_id == dept_id,
                                        Staff.qualification == "PhD").all()
    elif qual == "PhD":
        staff_list = Staff.query.filter_by(dept_id=dept_id, qualification="PhD").all()
    else:
        staff_list = Staff.query.filter_by(dept_id=dept_id).order_by(Staff.qualification).all()

    stu_query = Student.query.filter_by(dept_id=dept_id)
    if section != "all":
        stu_query = stu_query.filter_by(section=section)
    students = stu_query.order_by(Student.section, Student.roll_no).all()

    hod = HOD.query.filter_by(dept_id=dept_id).first()
    counts = {
        "BE":  Staff.query.filter_by(dept_id=dept_id, qualification="BE").count(),
        "ME":  Staff.query.filter_by(dept_id=dept_id, qualification="ME").count(),
        "PhD": Staff.query.filter_by(dept_id=dept_id, qualification="PhD").count(),
    }
    return render_template("principal/dept_view.html",
                           principal=principal, dept=dept,
                           staff_list=staff_list, students=students,
                           hod=hod, qual=qual, section=section, counts=counts)


@principal_bp.route("/staff")
@login_required
@principal_required
def all_staff():
    principal = current_user
    qual      = request.args.get("qual", "all")
    dept_id   = request.args.get("dept_id", "all")

    query = Staff.query
    if qual == "BE":
        query = query.filter(Staff.qualification.in_(["ME", "PhD"]))
    elif qual == "ME":
        query = query.filter(Staff.qualification == "PhD")
    elif qual == "PhD":
        query = query.filter_by(qualification="PhD")
    if dept_id != "all":
        query = query.filter_by(dept_id=int(dept_id))

    staff_list  = query.order_by(Staff.dept_id, Staff.qualification, Staff.name).all()
    departments = Department.query.all()
    return render_template("principal/staff.html",
                           principal=principal, staff_list=staff_list,
                           departments=departments, qual=qual, dept_id=dept_id)


@principal_bp.route("/hods")
@login_required
@principal_required
def hods():
    principal   = current_user
    hod_list    = HOD.query.order_by(HOD.id).all()
    departments = Department.query.all()
    dept_map    = {d.id: d.name for d in departments}
    return render_template("principal/hods.html",
                           principal=principal, hod_list=hod_list, dept_map=dept_map)


@principal_bp.route("/announce", methods=["GET", "POST"])
@login_required
@principal_required
def announce():
    principal = current_user
    if request.method == "POST":
        title   = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title and content:
            ann = Announcement(
                posted_by_role="principal",
                posted_by_id=principal.id,
                posted_by_name=principal.name,
                dept_id=None,
                title=title,
                content=content,
            )
            db.session.add(ann)
            db.session.commit()
            flash("College-wide announcement posted!", "success")
        else:
            flash("Title and content required.", "danger")
        return redirect(url_for("principal.announce"))

    announcements = (Announcement.query
                     .filter_by(posted_by_role="principal")
                     .order_by(Announcement.created_at.desc()).limit(20).all())
    return render_template("principal/announce.html",
                           principal=principal, announcements=announcements)


@principal_bp.route("/cameras")
@login_required
@principal_required
def cameras():
    principal   = current_user
    departments = Department.query.all()
    dept_id     = request.args.get("dept_id", "all")
    query       = CameraAlert.query
    if dept_id != "all":
        query = query.filter_by(dept_id=int(dept_id))
    alerts = query.order_by(CameraAlert.created_at.desc()).limit(50).all()
    return render_template("principal/cameras.html",
                           principal=principal, departments=departments,
                           alerts=alerts, dept_id=dept_id)


@principal_bp.route("/alerts/ack/<int:alert_id>", methods=["POST"])
@login_required
@principal_required
def ack_alert(alert_id):
    alert = CameraAlert.query.get_or_404(alert_id)
    alert.acknowledged = True
    db.session.commit()
    return jsonify({"status": "ok"})


@principal_bp.route("/contact-hod", methods=["GET", "POST"])
@login_required
@principal_required
def contact_hod():
    principal   = current_user
    departments = Department.query.all()
    hods        = HOD.query.filter(HOD.dept_id.isnot(None)).all()
    msg_sent    = False

    if request.method == "POST":
        hod_id  = request.form.get("hod_id")
        message = request.form.get("message", "").strip()
        if hod_id and message:
            hod = HOD.query.get(hod_id)
            if hod:
                # Store as an announcement from principal to that dept
                ann = Announcement(
                    posted_by_role="principal",
                    posted_by_id=principal.id,
                    posted_by_name=f"Principal {principal.name} → HOD",
                    dept_id=hod.dept_id,
                    title="Message from Principal",
                    content=message,
                )
                db.session.add(ann)
                db.session.commit()
                msg_sent = True
                flash(f"Message sent to HOD {hod.name}!", "success")

    return render_template("principal/contact_hod.html",
                           principal=principal, hods=hods,
                           departments=departments, msg_sent=msg_sent)


@principal_bp.route("/api/notifications/count")
@login_required
@principal_required
def notif_count():
    count = CameraAlert.query.filter_by(acknowledged=False).count()
    return jsonify({"count": count})
