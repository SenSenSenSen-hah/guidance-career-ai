import streamlit as st
import numpy as np
import json
import os
import threading
import hashlib
import requests
import sqlite3
import re
import google.generativeai as genai
import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 0. KONFIGURASI AWAL
# ==============================================================================
st.set_page_config(page_title="Autonomous Career AI", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; font-weight: 800; margin-bottom: 20px; }
    .step-card { background-color: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 25px; border-top: 5px solid #1f77b4; }
    .reasoning-text { font-style: italic; color: #444; background-color: #f0f7ff; padding: 20px; border-radius: 8px; border-left: 5px solid #1f77b4; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. ENGINE KECERDASAN BUATAN (AI)
# ==============================================================================

@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

nlp_model = load_nlp_model()

class GeminiCareerAnalyst:
    def __init__(self):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            st.warning(f"Gemini tidak tersedia: {e}")
            self.model = None

    def generate_personalized_insight(self, user_data, major_name, match_score):
        if not self.model:
            return "Analisis AI tidak tersedia. Periksa API Key."
        
        # IMPROVEMENT: Penambahan instruksi eksplisit agar menghindari karakter Unicode
        prompt = f"""
        Anda adalah Konselor Karir Profesional. Berikan analisis mendalam mengapa jurusan {major_name} 
        sangat cocok untuk siswa bernama {user_data.get('name')} berdasarkan profil berikut:
        - Jalur Sekolah: {user_data.get('stream')}
        - Nilai Rapor: {user_data.get('academic_scores')}
        - Esai Motivasi: "{user_data.get('essay')}"
        
        Instruksi:
        1. Gunakan Bahasa Indonesia yang akademis namun memotivasi.
        2. Hubungkan secara logis antara nilai mata pelajaran spesifik dengan tuntutan jurusan {major_name}.
        3. Jadikan "Esai Motivasi" sebagai landasan utama penentu minat/bakat.
        4. Berikan proyeksi karir masa depan.
        5. Maksimal 3 paragraf. Jangan tampilkan angka skor kecocokan dalam teks.
        6. PENTING: Gunakan teks biasa (plain text). HINDARI penggunaan bullet points khusus, tanda kutip miring (smart quotes), atau simbol non-standar agar laporan dapat dicetak ke PDF dengan aman. Gunakan tanda strip (-) untuk daftar.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gagal terhubung ke AI. Detail Error: {str(e)}"

class AdvancedCareerAI:
    def __init__(self, kb):
        self.kb = kb

    def _normalize(self, val, max_v=100):
        return min(max((val or 0) / max_v, 0), 1)

    def construct_user_radar_vector(self, user_data):
        sc = user_data.get('academic_scores', {})
        strm = user_data.get('stream', 'MIPA (IPA)')

        mw = sc.get('Math_W', 0)
        ind = sc.get('Indo', 0)
        ing = sc.get('Inggris', 0)

        # KORELASI RIASEC UNTUK SKRIPSI:
        # s_logika (Math) -> Investigative (I) / Conventional (C)
        # s_sosial (Social) -> Social (S)
        # s_verbal (Verbal) -> Enterprising (E) / Social (S)
        # s_seni (Art) -> Artistic (A)
        # s_sains (Science) -> Realistic (R) / Investigative (I)
        
        if strm == "MIPA (IPA)":
            s_logika = (mw * 0.3) + (sc.get('Math_M', 0) * 0.4) + (sc.get('Fisika', 0) * 0.3)
            s_sosial = (ind + ing) / 2.0
            s_sains = (sc.get('Fisika', 0) + sc.get('Kimia', 0) + sc.get('Biologi', 0)) / 3.0
            s_verbal = (ind + ing) / 2.0
            s_seni = (ind + ing) / 2.0 * 0.8
        elif strm == "IPS":
            s_logika = (mw * 0.5) + (sc.get('Ekonomi', 0) * 0.5)
            s_sosial = (sc.get('Sosiologi', 0) * 0.4) + (sc.get('Sejarah', 0) * 0.3) + (sc.get('Geografi', 0) * 0.3)
            s_sains = (sc.get('Geografi', 0) * 0.7) + (mw * 0.3)
            s_verbal = (ind + ing) / 2.0
            s_seni = (sc.get('Sejarah', 0) * 0.5) + (ind * 0.5)
        else:  # BAHASA
            s_logika = mw * 0.9
            s_sosial = (sc.get('Antropologi', 0) * 0.6) + (ind * 0.4)
            s_sains = mw * 0.7
            s_verbal = (ind * 0.3) + (ing * 0.3) + (sc.get('Sastra', 0) * 0.2) + (sc.get('Asing', 0) * 0.2)
            s_seni = (sc.get('Sastra', 0) * 0.7) + (ind * 0.3)

        v_math = self._normalize(s_logika)
        v_verbal = self._normalize(s_verbal)
        v_social = self._normalize(s_sosial)
        v_art = self._normalize(s_seni)
        v_science = self._normalize(s_sains)

        return np.array([v_math, v_verbal, v_social, v_art, v_science])

    def generate_recommendations(self, user_data):
        user_radar = self.construct_user_radar_vector(user_data).reshape(1, -1)
        user_essay = nlp_model.encode(user_data.get('essay', '')).reshape(1, -1)

        majors = self.kb.get_all_majors()
        results = []
        for name, data in majors.items():
            sim_acad = cosine_similarity(user_radar, np.array(data['radar_vector']).reshape(1, -1))[0][0]
            sim_sem = cosine_similarity(user_essay, data['semantic_vector'].reshape(1, -1))[0][0]
            
            # Bobot: 50% Rapor, 50% NLP Esai
            score = (sim_acad * 0.5 + sim_sem * 0.5) * 100
            results.append({
                'major': name,
                'score': round(score, 1),
                'vector': data['radar_vector'],
                'user_vector': user_radar[0].tolist()
            })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]


# ==============================================================================
# 2. DATABASE & SCRAPER
# ==============================================================================

_db_lock = threading.Lock()

def _fetch_wikipedia_sync(majors_list):
    results = []
    for m in majors_list:
        url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{m.replace(' ', '_')}"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                desc = r.json().get('extract', f"Program studi {m}")
            else:
                desc = f"Studi {m}"
        except Exception:
            desc = f"Studi {m}"
        results.append((m, desc))
    return results

class SQLiteKnowledgeBase:
    def __init__(self):
        self.conn = sqlite3.connect('knowledge_base.db', check_same_thread=False)
        self.conn.execute(
            'CREATE TABLE IF NOT EXISTS majors '
            '(major_name TEXT PRIMARY KEY, description TEXT, radar_vector TEXT, semantic_vector TEXT)'
        )
        if self.conn.execute("SELECT COUNT(*) FROM majors").fetchone()[0] == 0:
            self.process_and_save_new_majors([
                'Matematika', 'Teknik Informatika', 'Psikologi',
                'Manajemen', 'Arkeologi', 'Ilmu Komunikasi'
            ])

    def save_major(self, name, desc, radar_vec, semantic_vec):
        with _db_lock:
            self.conn.execute(
                'INSERT OR REPLACE INTO majors VALUES (?, ?, ?, ?)',
                (name, desc, json.dumps(radar_vec), json.dumps(semantic_vec.tolist()))
            )
            self.conn.commit()

    def get_all_majors(self):
        res = {}
        with _db_lock:
            rows = self.conn.execute("SELECT * FROM majors").fetchall()
        for r in rows:
            res[r[0]] = {
                'description': r[1],
                'radar_vector': json.loads(r[2]),
                'semantic_vector': np.array(json.loads(r[3]))
            }
        return res

    def process_and_save_new_majors(self, majors_list):
        data = _fetch_wikipedia_sync(majors_list)
        for m, desc in data:
            v = [0.1] * 5
            d_lower = desc.lower()
            if any(x in d_lower for x in ['hitung', 'logika', 'matematika', 'teknik', 'komputer']): v[0] = 0.9
            if any(x in d_lower for x in ['bahasa', 'komunikasi', 'sastra', 'tulis']): v[1] = 0.9
            if any(x in d_lower for x in ['sosial', 'masyarakat', 'manusia', 'hukum']): v[2] = 0.9
            if any(x in d_lower for x in ['seni', 'kreatif', 'desain', 'budaya']): v[3] = 0.9
            if any(x in d_lower for x in ['fisika', 'biologi', 'alam', 'medis', 'kimia']): v[4] = 0.9
            self.save_major(m, desc, v, nlp_model.encode(desc))

@st.cache_resource
def get_kb():
    return SQLiteKnowledgeBase()


# ==============================================================================
# 3. ANTARMUKA PENGGUNA (UI) & PEMBUATAN PDF
# ==============================================================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'LAPORAN REKOMENDASI KARIR AI', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)

# IMPROVEMENT: Fungsi pembersih teks agar PDF tidak crash karena karakter khusus Gemini
def sanitize_text_for_pdf(text):
    # Mengganti karakter kutip pintar
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    # Mengganti em-dash dan en-dash
    text = re.sub(r'[—–]', '-', text)
    # Mengganti bullet points Unicode
    text = re.sub(r'[•·]', '-', text)
    # Menghapus emoji atau karakter non-latin-1 lainnya secara paksa
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text

def render_step_1():
    st.markdown('<div class="step-card"><h3>Langkah 1: Identitas</h3>', unsafe_allow_html=True)
    with st.form("s1"):
        n = st.text_input("Nama Lengkap")
        s = st.selectbox("Peminatan Sekolah", ["MIPA (IPA)", "IPS", "Bahasa"])
        if st.form_submit_button("Lanjut"):
            if n:
                st.session_state.user_data.update({'name': n, 'stream': s})
                st.session_state.current_step = 1
                st.rerun()
            else:
                st.error("Mohon isi nama lengkap Anda.")
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_2():
    st.markdown('<div class="step-card"><h3>Langkah 2: Nilai Rapor</h3>', unsafe_allow_html=True)
    strm = st.session_state.user_data.get('stream')
    with st.form("s2"):
        c1, c2, c3 = st.columns(3)
        with c1:
            mw = st.number_input("Matematika (W)", 0, 100, 80)
            ind = st.number_input("B. Indo", 0, 100, 80)
        with c2:
            ing = st.number_input("B. Inggris", 0, 100, 80)

        if strm == "MIPA (IPA)":
            with c2: mm = st.number_input("Matematika (M)", 0, 100, 80)
            with c3:
                f = st.number_input("Fisika", 0, 100, 80)
                k = st.number_input("Kimia", 0, 100, 80)
                b = st.number_input("Biologi", 0, 100, 80)
            sc = {'Math_W': mw, 'Indo': ind, 'Inggris': ing, 'Math_M': mm, 'Fisika': f, 'Kimia': k, 'Biologi': b}
        elif strm == "IPS":
            with c2: ek = st.number_input("Ekonomi", 0, 100, 80)
            with c3:
                so = st.number_input("Sosiologi", 0, 100, 80)
                ge = st.number_input("Geografi", 0, 100, 80)
                sj = st.number_input("Sejarah", 0, 100, 80)
            sc = {'Math_W': mw, 'Indo': ind, 'Inggris': ing, 'Ekonomi': ek, 'Sosiologi': so, 'Geografi': ge, 'Sejarah': sj}
        else:  # BAHASA
            with c2: sas = st.number_input("Sastra Indo", 0, 100, 80)
            with c3:
                ant = st.number_input("Antropologi", 0, 100, 80)
                asg = st.number_input("B. Asing", 0, 100, 80)
            sc = {'Math_W': mw, 'Indo': ind, 'Inggris': ing, 'Sastra': sas, 'Antropologi': ant, 'Asing': asg}

        if st.form_submit_button("Lanjut"):
            st.session_state.user_data['academic_scores'] = sc
            st.session_state.current_step = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_3():
    st.markdown('<div class="step-card"><h3>Langkah 3: Esai Motivasi & Minat</h3>', unsafe_allow_html=True)
    st.info("Ceritakan hobi, pelajaran yang paling Anda nikmati, dan bayangan pekerjaan Anda di masa depan. AI akan menganalisis minat Anda dari cerita ini.")
    with st.form("s3"):
        es = st.text_area("Ceritakan di sini...", height=200)
        if st.form_submit_button("Analisis Hasil"):
            if len(es.strip().split()) < 10:
                st.error("Esai terlalu pendek. Minimal 10 kata.")
            else:
                st.session_state.user_data['essay'] = es
                st.session_state.current_step = 3
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_results():
    st.markdown('<h1 class="main-header">Hasil Rekomendasi Karir</h1>', unsafe_allow_html=True)
    kb = get_kb()
    ai = AdvancedCareerAI(kb)
    recs = ai.generate_recommendations(st.session_state.user_data)
    top_3 = recs[:3]

    fig = go.Figure()
    lbls = ['Logika', 'Verbal', 'Sosial', 'Seni', 'Sains']
    fig.add_trace(go.Scatterpolar(r=top_3[0]['user_vector'], theta=lbls, fill='toself', name='Kapasitas Rapor Anda'))
    fig.add_trace(go.Scatterpolar(r=top_3[0]['vector'], theta=lbls, name=top_3[0]['major']))
    st.plotly_chart(fig, use_container_width=True)

    ga = GeminiCareerAnalyst()
    tabs = st.tabs([f"1. {t['major']}" for t in top_3])

    user_hash = hashlib.md5(
        json.dumps(st.session_state.user_data, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"Skor Kecocokan: {top_3[i]['score']}%")
            key = f"insight_{i}_{user_hash}"
            if key not in st.session_state:
                with st.spinner("Gemini sedang menganalisis rapor dan esai..."):
                    st.session_state[key] = ga.generate_personalized_insight(
                        st.session_state.user_data, top_3[i]['major'], top_3[i]['score']
                    )
            st.markdown(f'<div class="reasoning-text">{st.session_state[key]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Membuat Dokumen PDF
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Nama: {st.session_state.user_data.get('name')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, f"Jurusan SMA: {st.session_state.user_data.get('stream')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    for i, r in enumerate(top_3):
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, f"{i+1}. {r['major']} ({r['score']}%)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=11)
        
        # Mengambil insight dan membersihkannya sebelum dimasukkan ke PDF
        raw_txt = st.session_state.get(f"insight_{i}_{user_hash}", "")
        safe_txt = sanitize_text_for_pdf(raw_txt)
        
        try:
            pdf.multi_cell(0, 7, safe_txt)
        except Exception as e:
            pdf.multi_cell(0, 7, "[Error: Teks mengandung format yang tidak didukung untuk dicetak]")
            st.error(f"Peringatan PDF: {e}")
            
        pdf.ln(5)

    pdf_bytes = pdf.output()
    st.download_button(
        label="📥 Unduh Hasil PDF",
        data=pdf_bytes,
        file_name=f"Rekomendasi_{st.session_state.user_data.get('name')}.pdf",
        mime="application/pdf"
    )

    if st.button("Ulangi Sesi"):
        st.session_state.clear()
        st.rerun()


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

def main():
    st.session_state.setdefault('user_data', {})
    st.session_state.setdefault('current_step', 0)

    with st.sidebar:
        st.markdown("<br>" * 15, unsafe_allow_html=True)
        with st.expander("⚙️"):
            if not st.session_state.get('admin', False):
                pw = st.text_input("Password", type="password")
                if st.button("Masuk"):
                    if "admin_password" in st.secrets and pw == st.secrets["admin_password"]:
                        st.session_state.admin = True
                        st.rerun()
                    else:
                        st.error("Password salah atau tidak diatur.")
            else:
                st.caption("Mode Admin Aktif")
                target = st.text_input("Pelajari Jurusan Baru (Wikipedia)")
                if st.button("Pelajari"):
                    get_kb().process_and_save_new_majors([target.title()])
                    st.success(f"{target} disimpan!")
                if st.button("Logout"):
                    st.session_state.admin = False
                    st.rerun()

    st.progress(st.session_state.current_step / 3)

    steps = [render_step_1, render_step_2, render_step_3, render_results]
    steps[st.session_state.current_step]()

if __name__ == "__main__":
    main()
