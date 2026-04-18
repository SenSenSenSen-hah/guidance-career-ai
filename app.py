import streamlit as st
import numpy as np
import json
import os
import base64
from datetime import datetime
from fpdf import FPDF
import sqlite3
import asyncio
import aiohttp
from bs4 import BeautifulSoup

# Library Machine Learning & AI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import google.generativeai as genai

st.set_page_config(page_title="Sistem Rekomendasi Karir AI", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# ==================== BLOK 1: PUSAT KECERDASAN BUATAN (AI) ====================
# ==============================================================================

# 1A. INISIALISASI MODEL NLP (Ekstraksi Esai Lokal)
@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

nlp_model = load_nlp_model()

# 1B. KONFIGURASI GEMINI LLM (Pakar Penjelasan)
class GeminiCareerAnalyst:
    def __init__(self):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except KeyError:
            st.error("Peringatan: GEMINI_API_KEY tidak ditemukan di secrets.toml.")
            self.model = None

    def generate_personalized_insight(self, user_data, major_name, match_score):
        if not self.model: return "Analisis mendalam AI tidak tersedia. Periksa API Key."

        name = user_data.get('name', 'Siswa')
        essay = user_data.get('essay', '')
        scores = user_data.get('academic_scores', {})
        
        prompt = f"""
        Anda adalah seorang Konselor Karir dan Pakar Pendidikan tingkat lanjut.
        Berdasarkan profil siswa berikut, jelaskan secara ilmiah dan analitis mengapa program studi {major_name} sangat relevan untuk mereka.
        (Skor Kecocokan Sistem: {match_score}%).

        Data Siswa:
        - Nama: {name}
        - Nilai Akademik: {scores}
        - Esai Personal: "{essay}"

        Instruksi Jawaban:
        1. Gunakan bahasa Indonesia yang profesional, memotivasi, dan akademis.
        2. Jangan menyebutkan metrik "Skor Kecocokan" secara eksplisit dalam teks.
        3. Buat korelasi logis antara nilai akademik atau narasi esai mereka dengan kompetensi yang dibangun di jurusan {major_name}.
        4. Berikan satu proyeksi karir spesifik yang sejalan dengan cita-cita di esai mereka.
        5. Batasi jawaban maksimal 3 paragraf padat.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception:
            return f"Berdasarkan analisis nilai dan minat, profil Anda memiliki keselarasan dengan kebutuhan {major_name}."

# 1C. LOGIKA PEMBOBOTAN RADAR (Teori RIASEC / Minat Bakat)
class HeuristicRadarGenerator:
    def __init__(self):
        self.keywords = {
            'math': ['matematika', 'hitung', 'logika', 'komputasi', 'algoritma', 'analisis data', 'angka', 'teknik'],
            'verbal': ['bahasa', 'sastra', 'tulis', 'baca', 'komunikasi', 'jurnalistik', 'sejarah', 'diplomasi'],
            'social': ['sosial', 'masyarakat', 'manusia', 'hukum', 'politik', 'psikologi', 'manajemen', 'bisnis'],
            'art': ['seni', 'desain', 'gambar', 'musik', 'kreatif', 'film', 'arsitektur'],
            'science': ['fisika', 'kimia', 'biologi', 'medis', 'dokter', 'farmasi', 'kesehatan']
        }

    def generate_vector(self, text, major_name):
        text, name = text.lower(), major_name.lower()
        scores = {'math': 0.1, 'verbal': 0.1, 'social': 0.1, 'art': 0.1, 'science': 0.1}
        
        if any(x in name for x in ['matematika', 'statistika', 'aktuaria']):
            scores['math'] = 0.99; scores['science'] = 0.4
        elif any(x in name for x in ['teknik', 'rekayasa', 'insinyur']):
            scores['math'] = 0.7; scores['science'] = 0.7
            if 'informatika' in name or 'komputer' in name: scores['math'] = 0.9; scores['science'] = 0.3
        elif any(x in name for x in ['psikologi', 'bimbingan']):
            scores['social'] = 0.95; scores['verbal'] = 0.7; scores['math'] = 0.2
        elif any(x in name for x in ['ekonomi', 'akuntansi', 'manajemen', 'bisnis']):
            scores['social'] = 0.7; scores['math'] = 0.6; scores['verbal'] = 0.4

        for dim, keys in self.keywords.items():
            count = sum(1 for key in keys if key in text)
            scores[dim] += min(count * 0.1, 0.3)

        return [round(max(min(scores[dim], 1.0), 0.1), 2) for dim in ['math', 'verbal', 'social', 'art', 'science']]

# 1D. PROSPEK KARIR & PENGEMBANGAN DIRI
class CareerInsightGenerator:
    def generate_insights(self, major_name):
        name = major_name.lower()
        careers = ["Praktisi Profesional", "Akademisi/Peneliti", "Wirausahawan"]
        develop = ["Soft Skill Komunikasi", "Manajemen Waktu", "Literasi Digital"]
        
        if 'teknik' in name: careers = ["Engineer", "Project Manager", "Konsultan Teknik"]; develop = ["Sertifikasi Insinyur", "Manajemen Proyek"]
        elif 'komputer' in name or 'informatika' in name: careers = ["Software Developer", "Data Analyst", "Cyber Security"]; develop = ["Algoritma", "Portofolio GitHub"]
        elif 'psikologi' in name: careers = ["HRD", "Psikolog", "Konselor"]; develop = ["Empati", "Asesmen Psikologi"]
            
        return careers, develop

# 1E. MESIN PENGGABUNG & PENGHITUNG KECOCOKAN (Algoritma Utama)
class AdvancedCareerAI:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.insight_gen = CareerInsightGenerator()

    def _normalize_score(self, val, max_val=100): return min(max((val or 0) / max_val, 0), 1)

    def construct_user_radar_vector(self, user_data):
        academics = user_data.get('academic_scores', {})
        interests = user_data.get('interests', {})
        competencies = user_data.get('competencies', {})
        
        score_logika = (academics.get('Matematika (Wajib)', 0) * 0.3) + (academics.get('Matematika (Peminatan)', 0) * 0.5) + (academics.get('Fisika', 0) * 0.2)
        vec_math = self._normalize_score(score_logika)
        
        avg_bahasa = (academics.get('Bahasa Indonesia', 0) + academics.get('Bahasa Inggris', 0)) / 2.0
        vec_verbal = (self._normalize_score(avg_bahasa) * 0.6) + (self._normalize_score(competencies.get('Komunikasi', 0), 5) * 0.4)
        
        vec_social = (self._normalize_score(academics.get('Sosiologi', 0)) * 0.4) + (self._normalize_score(interests.get('Komunikasi dan Sosial', 0), 5) * 0.4) + (self._normalize_score(competencies.get('Kepemimpinan', 0), 5) * 0.2)
        vec_art = (self._normalize_score(interests.get('Kreativitas dan Seni', 0), 5) * 0.6) + (self._normalize_score(competencies.get('Kreativitas', 0), 5) * 0.4)
        
        score_sains = (academics.get('Fisika', 0) + academics.get('Kimia', 0) + academics.get('Biologi', 0)) / 3.0
        vec_science = self._normalize_score(score_sains)
        
        return np.array([vec_math, vec_verbal, vec_social, vec_art, vec_science])

    def generate_recommendations(self, user_data):
        user_radar_vector = self.construct_user_radar_vector(user_data).reshape(1, -1)
        user_essay = user_data.get('essay', '')
        user_semantic_vector = nlp_model.encode(user_essay).reshape(1, -1) if user_essay else None

        all_majors = self.kb.get_all_majors()
        results = []
        
        for major_name, data in all_majors.items():
            major_radar_vec = np.array(data['radar_vector']).reshape(1, -1)
            academic_sim = cosine_similarity(user_radar_vector, major_radar_vec)[0][0]
            
            semantic_sim = 0
            if user_semantic_vector is not None:
                major_semantic_vec = data['semantic_vector'].reshape(1, -1)
                semantic_sim = cosine_similarity(user_semantic_vector, major_semantic_vec)[0][0]
            
            match_score = (academic_sim * 0.7 + semantic_sim * 0.3) * 100
            careers, develop = self.insight_gen.generate_insights(major_name)

            results.append({
                'major': major_name, 'score': round(min(match_score, 99.9), 1),
                'vector': data['radar_vector'], 'user_vector': user_radar_vector[0].tolist(),
                'careers': careers, 'develop': develop
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

# ==============================================================================
# ================= BLOK 2: DATABASE & AGEN SCRAPER (SISTEM) ===================
# ==============================================================================

class AsyncKnowledgeGatherer:
    def __init__(self): self.headers = {'User-Agent': 'Mozilla/5.0'}

    async def fetch_summary(self, session, major_name):
        url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{major_name.replace(' ', '_')}"
        try:
            async with session.get(url, headers=self.headers, timeout=5) as response:
                if response.status == 200: return major_name, (await response.json()).get('extract', f"Program Studi S1 {major_name}.")
        except Exception: pass
        return major_name, f"Program Studi S1 {major_name}."

    async def scrape_all_concurrently(self, majors_list):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_summary(session, major) for major in majors_list]
            return dict(await asyncio.gather(*tasks))

class SQLiteKnowledgeBase:
    def __init__(self):
        self.conn = sqlite3.connect('knowledge_base.db', check_same_thread=False)
        self.radar_gen = HeuristicRadarGenerator()
        self.conn.execute('CREATE TABLE IF NOT EXISTS majors (major_name TEXT PRIMARY KEY, description TEXT, radar_vector TEXT, semantic_vector TEXT)')
        self.conn.commit()
        if self.conn.execute("SELECT COUNT(*) FROM majors").fetchone()[0] == 0:
            self.process_and_save_new_majors(['Matematika', 'Teknik Informatika', 'Psikologi', 'Manajemen', 'Ilmu Komunikasi'])

    def save_major(self, name, desc, radar_vec, semantic_vec):
        self.conn.execute('INSERT OR REPLACE INTO majors VALUES (?, ?, ?, ?)', (name, desc, json.dumps(radar_vec), json.dumps(semantic_vec.tolist())))
        self.conn.commit()
        
    def get_all_majors(self):
        results = {}
        for row in self.conn.execute("SELECT major_name, description, radar_vector, semantic_vector FROM majors").fetchall():
            results[row[0]] = {'description': row[1], 'radar_vector': json.loads(row[2]), 'semantic_vector': np.array(json.loads(row[3]))}
        return results

    def get_info(self, major_name):
        row = self.conn.execute("SELECT description, radar_vector FROM majors WHERE major_name=?", (major_name,)).fetchone()
        return {'description': row[0], 'radar_vector': json.loads(row[1])} if row else {'description': "Data tidak tersedia.", 'radar_vector': [0.2]*5}

    def process_and_save_new_majors(self, majors_list):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        scraped_data = loop.run_until_complete(AsyncKnowledgeGatherer().scrape_all_concurrently(majors_list))
        for major_name, description in scraped_data.items():
            self.save_major(major_name, description, self.radar_gen.generate_vector(description, major_name), nlp_model.encode(description))

    def discover_new_majors(self):
        new_found = []
        blacklist = ["daftar", "kategori", "metode", "teori", "sejarah", "diploma", "magister", "doktor"]
        existing = list(self.get_all_majors().keys())
        for url in ["https://id.wikipedia.org/wiki/Kategori:Disiplin_akademik", "https://id.wikipedia.org/wiki/Kategori:Jurusan_pendidikan"]:
            try:
                import requests
                soup = BeautifulSoup(requests.get(url, timeout=5).content, 'html.parser')
                for link in soup.select("#mw-pages a"):
                    name = link.text.replace("Kategori:", "").strip()
                    if not any(bad in name.lower() for bad in blacklist) and name not in existing and 3 < len(name) < 50:
                        new_found.append(name); existing.append(name)
            except Exception: continue
        new_found = list(set(new_found))
        if new_found: self.process_and_save_new_majors(new_found)
        return new_found

@st.cache_resource
def get_knowledge_base(): return SQLiteKnowledgeBase()

# ==============================================================================
# ================== BLOK 3: ANTARMUKA PENGGUNA (STREAMLIT UI) =================
# ==============================================================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16); self.cell(0, 10, 'LAPORAN REKOMENDASI KARIR AI', 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .step-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 3px solid #4e54c8; }
    .info-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; border-left: 4px solid #2196f3; font-size: 0.9em; }
    .reasoning-text { font-style: italic; color: #555; background-color: #fff8e1; padding: 15px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

def render_step_1():
    st.markdown('<div class="step-card"><h3>Langkah 1: Identitas Diri</h3>', unsafe_allow_html=True)
    with st.form("step1"):
        name = st.text_input("Nama Lengkap")
        stream = st.selectbox("Peminatan Sekolah", ["MIPA (IPA)", "IPS", "Bahasa"])
        if st.form_submit_button("Lanjut"):
            if not name: st.error("Mohon isi nama lengkap.")
            else: st.session_state.user_data.update({'name': name, 'stream': stream}); st.session_state.current_step = 1; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_2():
    st.markdown('<div class="step-card"><h3>Langkah 2: Data Akademik</h3>', unsafe_allow_html=True)
    with st.form("step2"):
        c1, c2, c3 = st.columns(3)
        with c1: math_w = st.number_input("Matematika (Wajib)", 0, 100, 75); math_m = st.number_input("Matematika (Minat)", 0, 100, 75)
        with c2: indo = st.number_input("B. Indonesia", 0, 100, 75); fisika = st.number_input("Fisika", 0, 100, 75)
        with c3: inggris = st.number_input("B. Inggris", 0, 100, 75); sosio = st.number_input("Sosiologi", 0, 100, 75)
        if st.form_submit_button("Lanjut"):
            st.session_state.user_data['academic_scores'] = {'Matematika (Wajib)': math_w, 'Matematika (Peminatan)': math_m, 'Bahasa Indonesia': indo, 'Bahasa Inggris': inggris, 'Fisika': fisika, 'Sosiologi': sosio}
            st.session_state.current_step = 2; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_3():
    st.markdown('<div class="step-card"><h3>Langkah 3: Minat & Kompetensi</h3><div class="info-box">Skala 1 (Rendah) - 5 (Tinggi)</div><br>', unsafe_allow_html=True)
    with st.form("step3"):
        c1, c2 = st.columns(2)
        with c1: i1 = st.slider("Logika/Matematika", 1, 5, 3); i2 = st.slider("Sosial/Berdiskusi", 1, 5, 3)
        with c2: i3 = st.slider("Seni/Desain", 1, 5, 3); k1 = st.slider("Komunikasi", 1, 5, 3)
        if st.form_submit_button("Lanjut"):
            st.session_state.user_data['interests'] = {"Logika": i1, "Sosial": i2, "Seni": i3}
            st.session_state.user_data['competencies'] = {"Komunikasi": k1}
            st.session_state.current_step = 3; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_4():
    st.markdown('<div class="step-card"><h3>Langkah 4: Esai Personal</h3>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Ceritakan pelajaran favorit, hobi, dan cita-cita karir Anda dalam satu paragraf.</div><br>', unsafe_allow_html=True)
    with st.form("step4"):
        essay = st.text_area("Cerita Anda:", height=200, placeholder="Saya sangat suka pelajaran matematika dan komputer...")
        if st.form_submit_button("Analisis Hasil"):
            if len(essay.split()) < 10: st.error("Tuliskan minimal 10 kata agar AI dapat menganalisis kepribadian Anda.")
            else: st.session_state.user_data['essay'] = essay; st.session_state.current_step = 4; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_results_dashboard():
    st.markdown("## Hasil Rekomendasi Karir")
    kb = get_knowledge_base()
    with st.spinner("AI sedang memproses kecocokan..."):
        ai_engine = AdvancedCareerAI(kb)
        recommendations = ai_engine.generate_recommendations(st.session_state.user_data)
    
    top_3 = recommendations[:3]
    gemini_analyst = GeminiCareerAnalyst()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=top_3[0]['user_vector'], theta=['Logika', 'Verbal', 'Sosial', 'Seni', 'Sains'], fill='toself', name='Profil Anda'))
    fig.add_trace(go.Scatterpolar(r=top_3[0]['vector'], theta=['Logika', 'Verbal', 'Sosial', 'Seni', 'Sains'], fill='toself', name=top_3[0]['major'], opacity=0.5))
    st.plotly_chart(fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1]))), use_container_width=True)

    tabs = st.tabs([f"1: {top_3[0]['major']}", f"2: {top_3[1]['major']}", f"3: {top_3[2]['major']}"])
    for i, tab in enumerate(tabs):
        with tab:
            st.success(f"**Kecocokan: {top_3[i]['score']}%**")
            with st.spinner("Gemini sedang menyusun analisis pakar..."):
                if f"gemini_{i}" not in st.session_state:
                    st.session_state[f"gemini_{i}"] = gemini_analyst.generate_personalized_insight(st.session_state.user_data, top_3[i]['major'], top_3[i]['score'])
            st.markdown(f"<div class='reasoning-text'><b>Analisis Gemini LLM:</b><br>{st.session_state[f'gemini_{i}']}</div>", unsafe_allow_html=True)
            st.markdown(f"**Deskripsi:** {kb.get_info(top_3[i]['major'])['description']}")

    st.markdown("---")
    
    # KODE TOMBOL UNDUH PDF YANG SUDAH DIINTEGRASIKAN
    if st.button("📄 Download Laporan PDF"):
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "HASIL ANALISIS KARIR AI", 0, 1, 'C')
        pdf.ln(10)
        
        for i, rec in enumerate(top_3):
            if i > 0: pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.set_fill_color(230, 230, 250)
            pdf.cell(0, 10, f"PILIHAN {i+1}: {rec['major']} ({rec['score']}%)", 0, 1, 'L', 1)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, "Analisis Gemini LLM:", ln=1)
            pdf.set_font("Arial", '', 11)
            
            # Pengaman agar PDF tidak error jika tab belum diklik semua
            gemini_text = st.session_state.get(f"gemini_{i}", "Analisis AI sedang dimuat atau belum tersedia.")
            pdf.multi_cell(0, 6, gemini_text.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, "Prospek Karir & Pengembangan:", ln=1)
            pdf.set_font("Arial", '', 11)
            for c in rec['careers'] + rec['develop']: pdf.cell(0, 6, f"- {c}", ln=1)

        pdf_out = pdf.output(dest='S')
        
        # Konversi ke biner (bytes) jika bentuknya masih string
        if isinstance(pdf_out, str):
            pdf_out = pdf_out.encode('latin-1', 'replace')
            
        b64 = base64.b64encode(pdf_out).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Laporan_Karir_AI.pdf">Klik di sini untuk mengunduh PDF</a>', unsafe_allow_html=True)

    if st.button("🔄 Mulai Sesi Baru"): 
        st.session_state.clear()
        st.rerun()

def main():
    st.markdown('<h1 class="main-header">Autonomous Career AI</h1>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("Admin Panel")
        if st.button("Pindai Wikipedia"):
            with st.spinner("Scraping..."):
                res = get_knowledge_base().discover_new_majors()
                st.success(f"Ditemukan {len(res)} data.") if res else st.info("Tidak ada data baru.")
        
    if 'user_data' not in st.session_state: st.session_state.user_data = {}; st.session_state.current_step = 0
    st.progress((st.session_state.current_step + 1) / 5)
    
    [render_step_1, render_step_2, render_step_3, render_step_4, render_results_dashboard][st.session_state.current_step]()

if __name__ == "__main__":
    main()
