"""
seed.py — Populate the college_db with all initial data.
Run: python seed.py
"""
import sys
import random
import pymysql
from flask_bcrypt import generate_password_hash

# ── Connection ─────────────────────────────────────────────────────────────────
DB_CFG = dict(host="localhost", port=3306, user="root", password="root",
              charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)

DEPARTMENTS = ["IT", "CSE", "AIDS", "ECE", "EEE", "CIVIL"]
SECTIONS    = ["A", "B"]
SUBJECTS    = ["English", "Physics", "Chemistry", "Maths", "Python", "Tamil"]

# Staff per dept: BE/ME/PhD breakdown (total 287)
# IT=55, CSE=55, AIDS=50, ECE=50, EEE=42, CIVIL=35
STAFF_DEPT_CFG = {
    "IT":    {"BE": 25, "ME": 20, "PhD": 10},
    "CSE":   {"BE": 25, "ME": 20, "PhD": 10},
    "AIDS":  {"BE": 22, "ME": 18, "PhD": 10},
    "ECE":   {"BE": 22, "ME": 18, "PhD": 10},
    "EEE":   {"BE": 19, "ME": 14, "PhD":  9},
    "CIVIL": {"BE": 15, "ME": 13, "PhD":  7},
}

# Students per dept (CIVIL=100, others=200)
STUDENT_DEPT_CFG = {
    "IT": 200, "CSE": 200, "AIDS": 200, "ECE": 200, "EEE": 200, "CIVIL": 100
}

FIRST_NAMES = [
    "Aarav","Aditya","Akash","Ajay","Anil","Anish","Ankit","Arjun","Arnav","Ashok",
    "Bala","Bharath","Charan","Deepak","Dev","Dinesh","Ganesh","Gopal","Harish","Hari",
    "Ishaan","Jagadish","Karthik","Kavin","Krishna","Kumar","Lokesh","Mahesh","Mani","Mohan",
    "Naveen","Nikhil","Nilesh","Om","Pavan","Prakash","Pranav","Priya","Rahul","Raj",
    "Rajan","Rajesh","Ram","Ramesh","Ravi","Rohit","Sai","Sankar","Santhosh","Saran",
    "Sarath","Sathish","Selvam","Senthil","Shiva","Sibi","Siddarth","Siva","Suresh","Tamil",
    "Thiru","Uday","Vignesh","Vijay","Vikas","Vinod","Vishnu","Yuvan","Zaid","Surya",
    "Arun","Bala","Cheran","Deva","Elan","Faisal","Gokul","Harikumar","Indra","Jagan",
    "Kannan","Lakshmanan","Manoj","Nandha","Oviya","Padma","Raghu","Sakthi","Tamilarasan","Uma"
]
LAST_NAMES = [
    "Kumar","Raj","Rajan","Reddy","Sharma","Singh","Pillai","Nair","Rao","Iyer",
    "Krishnan","Murugan","Bala","Pandi","Selvan","Velu","Arumugam","Durai","Ganesan","Palani",
    "Subramanian","Sundaram","Ramachandran","Venkatesh","Annamalai","Chandrasekaran","Muthusamy","Natarajan","Perumal","Ramasamy"
]

def rnd_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def rnd_phone():
    return f"9{random.randint(100000000,999999999)}"

def rnd_email(name, idx):
    n = name.lower().replace(" ", ".")
    return f"{n}{idx}@college.edu.in"

def rnd_address():
    streets = ["Anna Salai","Gandhi Road","Nehru Street","Rajaji Road","Kamaraj Nagar"]
    cities  = ["Chennai","Coimbatore","Madurai","Trichy","Salem","Erode","Vellore","Tirunelveli"]
    return f"{random.randint(1,999)}, {random.choice(streets)}, {random.choice(cities)} - {random.randint(600001,643001)}"

def hash_pw(password: str) -> str:
    return generate_password_hash(password.encode(), 4).decode("utf-8")


def create_database():
    con = pymysql.connect(**DB_CFG)
    with con.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS college_db")
        cur.execute("CREATE DATABASE college_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    con.commit()
    con.close()
    print("[OK] Database college_db ensured.")


def create_tables(con):
    ddl = """
    CREATE TABLE IF NOT EXISTS departments (
        id   INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS subjects (
        id   INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS students (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        si_no         INT UNIQUE NOT NULL,
        roll_no       VARCHAR(20) UNIQUE NOT NULL,
        name          VARCHAR(100) NOT NULL,
        username      VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        dept_id       INT NOT NULL,
        section       ENUM('A','B') NOT NULL,
        phone         VARCHAR(15),
        email         VARCHAR(100),
        address       TEXT,
        photo         VARCHAR(255) DEFAULT 'default_student.png',
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    );

    CREATE TABLE IF NOT EXISTS staff (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        name          VARCHAR(100) NOT NULL,
        username      VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        dept_id       INT NOT NULL,
        qualification ENUM('BE','ME','PhD') NOT NULL,
        phone         VARCHAR(15),
        email         VARCHAR(100),
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    );

    CREATE TABLE IF NOT EXISTS hods (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        name          VARCHAR(100) NOT NULL,
        username      VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        dept_id       INT,
        phone         VARCHAR(15),
        email         VARCHAR(100),
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    );

    CREATE TABLE IF NOT EXISTS principals (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        name          VARCHAR(100) NOT NULL,
        username      VARCHAR(50) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        phone         VARCHAR(15),
        email         VARCHAR(100),
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS marks (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        student_id INT NOT NULL,
        subject_id INT NOT NULL,
        internal1  INT DEFAULT 0,
        internal2  INT DEFAULT 0,
        internal3  INT DEFAULT 0,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_student_subject (student_id, subject_id),
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
        FOREIGN KEY (subject_id) REFERENCES subjects(id)
    );

    CREATE TABLE IF NOT EXISTS attendance (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        student_id INT NOT NULL,
        date       DATE NOT NULL,
        status     ENUM('P','A') NOT NULL DEFAULT 'P',
        staff_id   INT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_student_date (student_id, date),
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
        FOREIGN KEY (staff_id) REFERENCES staff(id)
    );

    CREATE TABLE IF NOT EXISTS announcements (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        posted_by_role ENUM('staff','hod','principal') NOT NULL,
        posted_by_id   INT NOT NULL,
        posted_by_name VARCHAR(100),
        dept_id        INT,
        title          VARCHAR(200) NOT NULL,
        content        TEXT NOT NULL,
        created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    );

    CREATE TABLE IF NOT EXISTS camera_alerts (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        dept_id       INT NOT NULL,
        alert_type    VARCHAR(50) NOT NULL,
        snapshot_path VARCHAR(255),
        camera_index  INT DEFAULT 0,
        acknowledged  TINYINT(1) DEFAULT 0,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dept_id) REFERENCES departments(id)
    );
    """
    with con.cursor() as cur:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
    con.commit()
    print("[OK] All tables created.")


def seed_departments(con):
    with con.cursor() as cur:
        for dept in DEPARTMENTS:
            cur.execute("INSERT IGNORE INTO departments (name) VALUES (%s)", (dept,))
    con.commit()
    print("[OK] Departments seeded.")


def seed_subjects(con):
    with con.cursor() as cur:
        for subj in SUBJECTS:
            cur.execute("INSERT IGNORE INTO subjects (name) VALUES (%s)", (subj,))
    con.commit()
    print("[OK] Subjects seeded.")


def get_dept_map(con):
    with con.cursor() as cur:
        cur.execute("SELECT id, name FROM departments")
        return {row["name"]: row["id"] for row in cur.fetchall()}


def get_subject_ids(con):
    with con.cursor() as cur:
        cur.execute("SELECT id FROM subjects ORDER BY id")
        return [row["id"] for row in cur.fetchall()]


STUDENT_IMAGES = [
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1488161628813-04466f872be2?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1501196354995-cbb51c65aaea?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1534308983496-4fabb1a015ee?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1554151228-14d9def656e4?w=300&auto=format&fit=crop&q=80",
    "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=300&auto=format&fit=crop&q=80"
]

def seed_students(con):
    dept_map = get_dept_map(con)
    subject_ids = get_subject_ids(con)
    student_num = 1
    si_counter  = 1
    total = 0

    print("[..] Seeding students (this may take a minute)…")
    for dept_name, count in STUDENT_DEPT_CFG.items():
        dept_id = dept_map[dept_name]
        per_section = count // 2      # split evenly across A and B

        for section in SECTIONS:
            for _ in range(per_section):
                username  = f"student{str(student_num).zfill(3)}"
                roll_no   = f"{dept_name}{section}{str(student_num).zfill(4)}"
                password  = roll_no                     # password = roll number
                pw_hash   = hash_pw(password)
                name      = rnd_name()
                phone     = rnd_phone()
                email     = rnd_email(name, student_num)
                address   = rnd_address()
                photo_url = random.choice(STUDENT_IMAGES)

                with con.cursor() as cur:
                    cur.execute("""
                        INSERT IGNORE INTO students
                            (si_no, roll_no, name, username, password_hash,
                             dept_id, section, phone, email, address, photo)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (si_counter, roll_no, name, username, pw_hash,
                          dept_id, section, phone, email, address, photo_url))
                    new_id = cur.lastrowid

                # Seed marks for each subject
                if new_id:
                    with con.cursor() as cur:
                        for subj_id in subject_ids:
                            cur.execute("""
                                INSERT IGNORE INTO marks
                                    (student_id, subject_id, internal1, internal2, internal3)
                                VALUES (%s,%s,%s,%s,%s)
                            """, (new_id, subj_id,
                                  random.randint(20, 100),
                                  random.randint(20, 100),
                                  random.randint(20, 100)))

                student_num += 1
                si_counter  += 1
                total       += 1

                if total % 100 == 0:
                    con.commit()
                    print(f"    … {total} students inserted")

    con.commit()
    print(f"[OK] {total} students seeded.")



def seed_staff(con):
    dept_map = get_dept_map(con)
    staff_num = 1
    total = 0
    staff_pw_hash = hash_pw("Welcome@123")

    print("[..] Seeding staff…")
    for dept_name, cfg in STAFF_DEPT_CFG.items():
        dept_id = dept_map[dept_name]
        for qual, qty in cfg.items():
            for _ in range(qty):
                username = f"staff@{str(staff_num).zfill(2)}"
                name     = rnd_name()
                phone    = rnd_phone()
                email    = rnd_email(name, staff_num)

                with con.cursor() as cur:
                    cur.execute("""
                        INSERT IGNORE INTO staff
                            (name, username, password_hash, dept_id, qualification, phone, email)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """, (name, username, staff_pw_hash, dept_id, qual, phone, email))

                staff_num += 1
                total     += 1

    con.commit()
    print(f"[OK] {total} staff seeded.")


def seed_hods(con):
    dept_map = get_dept_map(con)
    hod_pw_hash = hash_pw("welcome@123")

    # HOD@01–06: one per dept; HOD@07–11: spare/assistant (no dept)
    hod_names = [
        "Dr. Rajesh Kumar",      # HOD@01 - IT
        "Dr. Priya Sharma",      # HOD@02 - CSE
        "Dr. Arun Selvam",       # HOD@03 - AIDS
        "Dr. Meena Krishnan",    # HOD@04 - ECE
        "Dr. Suresh Babu",       # HOD@05 - EEE
        "Dr. Kavitha Rajan",     # HOD@06 - CIVIL
        "Dr. Venkat Subramanian",# HOD@07
        "Dr. Sathish Pandian",   # HOD@08
        "Dr. Anitha Murugan",    # HOD@09
        "Dr. Bharath Durai",     # HOD@10
        "Dr. Lalitha Ganesan",   # HOD@11
    ]
    dept_names = DEPARTMENTS + [None, None, None, None, None]

    with con.cursor() as cur:
        for i, (name, dept_name) in enumerate(zip(hod_names, dept_names), start=1):
            username = f"HOD@{str(i).zfill(2)}"
            dept_id  = dept_map.get(dept_name) if dept_name else None
            phone    = rnd_phone()
            email    = f"hod.{dept_name.lower() if dept_name else 'admin'}{i}@college.edu.in"
            cur.execute("""
                INSERT IGNORE INTO hods
                    (name, username, password_hash, dept_id, phone, email)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (name, username, hod_pw_hash, dept_id, phone, email))

    con.commit()
    print("[OK] 11 HODs seeded.")


def seed_principals(con):
    principal_pw_hash = hash_pw("Principal@123")
    principals = [
        ("Dr. S. Mohanraj", "Principal_1", "9876543210", "principal1@college.edu.in"),
        ("Dr. R. Vijayalakshmi", "Principal_2", "9876543211", "principal2@college.edu.in"),
    ]
    with con.cursor() as cur:
        for name, username, phone, email in principals:
            cur.execute("""
                INSERT IGNORE INTO principals
                    (name, username, password_hash, phone, email)
                VALUES (%s,%s,%s,%s,%s)
            """, (name, username, principal_pw_hash, phone, email))
    con.commit()
    print("[OK] 2 Principals seeded.")


def seed_sample_announcements(con):
    dept_map = get_dept_map(con)
    with con.cursor() as cur:
        # Staff announcement (IT dept)
        cur.execute("""
            INSERT IGNORE INTO announcements
                (posted_by_role, posted_by_id, posted_by_name, dept_id, title, content)
            VALUES ('staff', 1, 'Staff Member', %s,
                    'Internal Test Schedule', 'Internal Test 1 for all subjects is scheduled for next Monday.')
        """, (dept_map["IT"],))

        # HOD announcement (CSE dept)
        cur.execute("""
            INSERT IGNORE INTO announcements
                (posted_by_role, posted_by_id, posted_by_name, dept_id, title, content)
            VALUES ('hod', 2, 'Dr. Priya Sharma', %s,
                    'Department Meeting', 'All CSE students must attend the department meeting on Friday at 10 AM.')
        """, (dept_map["CSE"],))

        # Principal announcement (all depts)
        cur.execute("""
            INSERT IGNORE INTO announcements
                (posted_by_role, posted_by_id, posted_by_name, dept_id, title, content)
            VALUES ('principal', 1, 'Dr. S. Mohanraj', NULL,
                    'College Annual Day', 'Annual Day celebrations will be held on 15th June. All students are invited.')
        """)
    con.commit()
    print("[OK] Sample announcements seeded.")


def seed_attendance(con):
    print("[..] Seeding attendance for the last 5 days…")
    import datetime
    
    # Get all students
    with con.cursor() as cur:
        cur.execute("SELECT id, dept_id, section FROM students")
        students = cur.fetchall()
        
    # Get staff by department
    with con.cursor() as cur:
        cur.execute("SELECT id, dept_id FROM staff")
        staff_list = cur.fetchall()
        
    staff_by_dept = {}
    for s in staff_list:
        staff_by_dept.setdefault(s["dept_id"], []).append(s["id"])
        
    # Generate past dates (excluding Sunday)
    dates = []
    curr = datetime.date.today()
    while len(dates) < 5:
        if curr.weekday() != 6:  # 6 is Sunday
            dates.append(curr)
        curr -= datetime.timedelta(days=1)
        
    total = 0
    with con.cursor() as cur:
        for dt in dates:
            for stu in students:
                # 90% Present, 10% Absent
                status = "P" if random.random() < 0.90 else "A"
                # Pick a random staff from same dept
                dept_staff = staff_by_dept.get(stu["dept_id"], [])
                staff_id = random.choice(dept_staff) if dept_staff else None
                
                cur.execute("""
                    INSERT IGNORE INTO attendance (student_id, date, status, staff_id)
                    VALUES (%s, %s, %s, %s)
                """, (stu["id"], dt, status, staff_id))
                total += 1
                
    con.commit()
    print(f"[OK] {total} attendance records seeded.")


def seed_camera_alerts(con):
    print("[..] Seeding camera alerts…")
    import datetime
    from pathlib import Path
    import shutil
    
    dept_map = get_dept_map(con)
    alert_types = ["fire", "fighting", "sleeping", "eating", "playing", "dancing", "person"]
    
    static_snapshots_dir = Path(__file__).parent / "static" / "snapshots"
    default_img_path = Path(__file__).parent / "static" / "img" / "default_student.png"
    
    total = 0
    with con.cursor() as cur:
        for dept_name, dept_id in dept_map.items():
            # Let's create a directory for this department
            dept_dir = static_snapshots_dir / dept_name
            dept_dir.mkdir(parents=True, exist_ok=True)
            
            # Let's add 2 alerts for each department
            for i in range(2):
                alert_type = random.choice(alert_types)
                camera_idx = random.randint(0, 2)
                acknowledged = 0 if i == 0 else 1  # 1 unread, 1 read
                
                # Snapshot filename and path
                filename = f"sample_{alert_type}_{i+1}.jpg"
                snapshot_path = f"snapshots/{dept_name}/{filename}"
                full_path = dept_dir / filename
                
                # Copy default_student.png to serve as a valid snapshot file
                if default_img_path.exists() and not full_path.exists():
                    shutil.copy(default_img_path, full_path)
                
                # Random timestamp in the last 2 days
                ts = datetime.datetime.now() - datetime.timedelta(
                    hours=random.randint(1, 48)
                )
                
                cur.execute("""
                    INSERT IGNORE INTO camera_alerts
                        (dept_id, alert_type, snapshot_path, camera_index, acknowledged, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (dept_id, alert_type, snapshot_path, camera_idx, acknowledged, ts))
                total += 1
                
    con.commit()
    print(f"[OK] {total} camera alerts seeded.")


def print_summary(con):
    tables = ["departments", "subjects", "students", "staff", "hods",
              "principals", "marks", "attendance", "announcements", "camera_alerts"]
    print("\n-- Row counts ------------------------------")
    with con.cursor() as cur:
        for t in tables:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {t}")
            cnt = cur.fetchone()["cnt"]
            print(f"  {t:<20} {cnt:>6}")
    print("--------------------------------------------\n")


if __name__ == "__main__":
    print("=== College Management System — Database Seeder ===\n")
    create_database()

    con = pymysql.connect(database="college_db", **DB_CFG)
    try:
        create_tables(con)
        seed_departments(con)
        seed_subjects(con)
        seed_students(con)
        seed_staff(con)
        seed_hods(con)
        seed_principals(con)
        seed_sample_announcements(con)
        seed_attendance(con)
        seed_camera_alerts(con)
        print_summary(con)
        print("[OK] Seeding complete! You can now run: flask run")
    except Exception as e:
        print(f"[ERR] Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        con.close()
