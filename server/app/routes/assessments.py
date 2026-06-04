from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models import Assessment, Question, User
from app.utils.certainty_factor import diagnose as cf_diagnose
from app.utils.ml_stress import predict_ml, combine_cf_ml, REKOMENDASI

assessments_bp = Blueprint("assessments", __name__)


def get_current_user_id():
    """Helper untuk mengambil user_id dari JWT, atau fallback ke default user jika opsional."""
    user_id = get_jwt_identity()
    if user_id:
        return int(user_id)
        
    # Fallback ke default user
    default_user = User.query.first()
    if not default_user:
        # Jika belum ada user sama sekali di DB, buat default user
        default_user = User(
            name="Karyawan Capstone",
            email="karyawan@capstone.com",
            password="mockpassword123" # password mock
        )
        db.session.add(default_user)
        db.session.commit()
    return default_user.id


@assessments_bp.route("/assessments", methods=["POST"])
@jwt_required(optional=True)
def submit_assessment():
    """
    Submit jawaban kuesioner dan dapatkan analisis stres hybrid (CF + ML).

    Header:
        Authorization: Bearer <JWT_TOKEN> (Opsional untuk testing/demo)

    Body JSON:
        {
            "answers": {
                "G1": 4,
                "G2": 3,
                ...
                "A1": 9.5,
                "A2": 1,
                ...
            }
        }

    Returns:
        201: Hasil analisis stres hybrid
        400: Validasi gagal
    """
    user_id = get_current_user_id()
    data = request.get_json()

    if not data or "answers" not in data:
        return jsonify({"error": "Jawaban kuesioner wajib diisi"}), 400

    raw_answers = data["answers"]

    # Validasi jawaban
    if not isinstance(raw_answers, dict) or len(raw_answers) == 0:
        return jsonify({"error": "Format jawaban tidak valid"}), 400

    # Ambil semua pertanyaan dari database untuk pemetaan ID ke Code
    questions = Question.query.all()
    if not questions:
        return jsonify({"error": "Belum ada pertanyaan di database. Jalankan seed terlebih dahulu."}), 500

    # Petakan key jawaban ke Question Code jika frontend mengirimkan ID database
    id_to_code = {str(q.id): q.code for q in questions}
    answers = {}
    for k, v in raw_answers.items():
        code = id_to_code.get(str(k), str(k))
        answers[code] = v

    # ─────────────────────────────────────────
    # 1. PARSE CERTAINTY FACTOR ANSWERS (G1-G43)
    # ─────────────────────────────────────────
    # Map jawaban 1-5 ke CF user 0.0-1.0
    cf_scale_map = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
    
    # 10 CF pertanyaan yang aktif
    active_cf_codes = ["G1", "G2", "G3", "G16", "G18", "G23", "G26", "G33", "G37", "G38"]
    cf_answers = {}
    
    for code in active_cf_codes:
        val = answers.get(code)
        if val is not None:
            # Jika user mengirim jawaban angka 1-5, petakan ke 0.0-1.0
            if isinstance(val, int) and val in cf_scale_map:
                cf_answers[code] = cf_scale_map[val]
            else:
                try:
                    cf_answers[code] = float(val)
                except ValueError:
                    cf_answers[code] = 0.0
        else:
            cf_answers[code] = 0.0
            
    # Default-kan pertanyaan G-lainnya (1-43) yang tidak ditanyakan menjadi 0.0
    for i in range(1, 44):
        g_code = f"G{i}"
        if g_code not in cf_answers:
            cf_answers[g_code] = 0.0

    # ─────────────────────────────────────────
    # 2. PARSE MACHINE LEARNING FEATURES
    # ─────────────────────────────────────────
    try:
        # A1: Avg_Working_Hours_Per_Day (float)
        hours = answers.get("A1")
        if hours is None:
            hours = 8.0
        else:
            hours = float(hours)
            
        ml_features = {
            "Avg_Working_Hours_Per_Day": hours,
            "Work_From": int(answers.get("A2", 1)),          # 0=Kantor, 1=WFH, 2=Hybrid
            "Lives_With_Family": int(answers.get("A3", 1)),  # 0=Tidak, 1=Ya
            "Social_Person": int(answers.get("A4", 3)),      # 1-5
            "Work_Life_Balance": int(answers.get("A5", 1)),  # 0=Tidak, 1=Ya
            "Sleeping_Habit": int(answers.get("ML1", 3)),    # 1-5
            "Exercise_Habit": int(answers.get("ML2", 3)),    # 1-5
            "Work_Pressure": int(answers.get("ML3", 3)),     # 1-5
            "Manager_Support": int(answers.get("ML4", 3)),   # 1-5
            "Job_Satisfaction": int(answers.get("ML5", 3))   # 1-5
        }
    except Exception as e:
        return jsonify({"error": f"Format data aktivitas harian (ML) tidak valid: {str(e)}"}), 400

    # ─────────────────────────────────────────
    # 3. JALANKAN DIAGNOSIS HYBRID (CF + ML)
    # ─────────────────────────────────────────
    try:
        # Jalankan CF
        cf_res = cf_diagnose(cf_answers)
        
        # Jalankan ML
        ml_res = predict_ml(ml_features)
        
        # Kombinasikan CF (bobot 0.7) dan ML (bobot 0.3)
        combined_res = combine_cf_ml(cf_res, ml_res, cf_weight=0.7)
        
        stress_level = combined_res["final_diagnosis"]
        final_code = combined_res["final_code"]
        
        # Hitung persentase tingkat stres yang konsisten dengan rentang kategori diagnosis:
        # D1: 0% - 25%, D2: 26% - 50%, D3: 51% - 75%, D4: 76% - 100%
        breakdown = combined_res["score_breakdown"]
        p_D1 = breakdown.get("Tidak Stres", 0.0)
        p_D2 = breakdown.get("Stres Ringan", 0.0)
        p_D3 = breakdown.get("Stres Sedang", 0.0)
        p_D4 = breakdown.get("Stres Berat", 0.0)
        
        final_idx = int(final_code[1]) - 1  # D1 -> 0, D2 -> 1, D3 -> 2, D4 -> 3
        
        if final_idx == 0:  # Tidak Stres (D1) -> Range [0, 25]
            calculated_score = 25 * (1.0 - p_D1)
        elif final_idx == 1:  # Stres Ringan (D2) -> Range [26, 50]
            denom = p_D1 + p_D3 + p_D4
            fraction = (p_D3 + p_D4) / denom if denom > 0 else 0.5
            calculated_score = 26 + fraction * 24
        elif final_idx == 2:  # Stres Sedang (D3) -> Range [51, 75]
            denom = p_D1 + p_D2 + p_D4
            fraction = p_D4 / denom if denom > 0 else 0.5
            calculated_score = 51 + fraction * 24
        elif final_idx == 3:  # Stres Berat (D4) -> Range [76, 100]
            calculated_score = 76 + ((p_D4 - 0.25) / 0.75) * 24
        else:
            calculated_score = 0.0
            
        score = int(round(calculated_score))

        
        # Ambil rekomendasi
        rec_info = REKOMENDASI.get(final_code, REKOMENDASI["D1"])
        recommendations = [rec_info["summary"]] + rec_info["tips"]
        
    except Exception as e:
        return jsonify({"error": f"Gagal menjalankan model AI: {str(e)}"}), 500

    # ─────────────────────────────────────────
    # 4. HITUNG ANALISIS FAKTOR PEMICU (CF)
    # ─────────────────────────────────────────
    category_questions = {
        "Beban dan Tekanan Kerja": ["G1", "G2", "G3"],
        "Konflik Peran dan Penugasan": ["G16"],
        "Hubungan Interpersonal di Tempat Kerja": ["G18"],
        "Kejelasan Peran dan Informasi Kerja": ["G23", "G26"],
        "Gaya Kepemimpinan dan Penilaian Kinerja": ["G33"],
        "Pengembangan Karir dan Kepuasan Kerja": ["G37", "G38"]
    }
    
    category_colors = {
        "Beban dan Tekanan Kerja": "bg-red-500",
        "Konflik Peran dan Penugasan": "bg-orange-500",
        "Hubungan Interpersonal di Tempat Kerja": "bg-yellow-500",
        "Kejelasan Peran dan Informasi Kerja": "bg-blue-500",
        "Gaya Kepemimpinan dan Penilaian Kinerja": "bg-indigo-500",
        "Pengembangan Karir dan Kepuasan Kerja": "bg-purple-500"
    }
    
    factors = []
    for cat_name, codes in category_questions.items():
        vals = [cf_answers[c] for c in codes if c in cf_answers]
        if vals:
            avg_val = sum(vals) / len(vals)
            percentage = round(avg_val * 100)
        else:
            percentage = 0
            
        factors.append({
            "label": cat_name,
            "value": percentage,
            "color": category_colors.get(cat_name, "bg-green-500")
        })

    # ─────────────────────────────────────────
    # 5. SIMPAN HASIL KE DATABASE
    # ─────────────────────────────────────────
    assessment = Assessment(
        user_id=user_id,
        score=score,
        stress_level=stress_level,
        factors=factors,
        recommendations=recommendations,
        answers=raw_answers, # simpan jawaban asli yang dikirim frontend
    )
    db.session.add(assessment)
    db.session.commit()

    return jsonify({
        "message": "Analisis stres berhasil dilakukan dengan model hybrid!",
        "result": assessment.to_dict(),
    }), 201


@assessments_bp.route("/stress-result", methods=["GET"])
@jwt_required(optional=True)
def get_stress_result():
    """
    Ambil hasil analisis stres terakhir dari user yang sedang login.

    Header:
        Authorization: Bearer <JWT_TOKEN> (Opsional)

    Returns:
        200: Hasil analisis stres terakhir
        404: Belum pernah melakukan tes
    """
    user_id = get_current_user_id()

    # Ambil assessment terbaru milik user ini
    assessment = (
        Assessment.query
        .filter_by(user_id=user_id)
        .order_by(Assessment.created_at.desc())
        .first()
    )

    if not assessment:
        return jsonify({"error": "Anda belum pernah melakukan tes. Silakan selesaikan kuesioner terlebih dahulu."}), 404

    return jsonify(assessment.to_dict()), 200


@assessments_bp.route("/assessments/history", methods=["GET"])
@jwt_required(optional=True)
def get_assessment_history():
    """
    Ambil semua riwayat tes stres milik user.

    Returns:
        200: List riwayat assessment
    """
    user_id = get_current_user_id()

    assessments = (
        Assessment.query
        .filter_by(user_id=user_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )

    return jsonify({
        "history": [a.to_dict() for a in assessments],
        "total": len(assessments),
    }), 200
