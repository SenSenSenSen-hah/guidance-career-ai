import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import threading
import time
import requests
import base64
from datetime import datetime
from fpdf import FPDF
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk

# Machine Learning & Visualization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ==================== 1. KONFIGURASI SISTEM ====================

def download_nltk_data():
    resources = ['punkt', 'averaged_perceptron_tagger', 'brown']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

download_nltk_data()

st.set_page_config(
    page_title="Sistem Pakar Karir SMA (AI Hybrid)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    .step-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-top: 3px solid #4e54c8;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        font-size: 0.9em;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 2. SISTEM PENGETAHUAN HYBRID ====================

class AutonomousKnowledgeBase:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.db_file = 'knowledge_base.json'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.data = self._load_db()
        self.is_running = False

    def _load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_db(self):
        with self._lock:
            with open(self.db_file, 'w') as f:
                json.dump(self.data, f, indent=4)

    def _scrape_wikipedia(self, major_name):
        try:
            query = major_name.replace(' ', '_')
            url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{query}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'description': data.get('extract', 'Tidak ada deskripsi.'),
                    'source': 'Wikipedia (Live Scrape)',
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '#'),
                    'timestamp': datetime.now().isoformat()
                }
        except:
            pass
        
        return {
            'description': f"Program studi {major_name} mempelajari aspek teoretis dan praktis di bidangnya. Lulusan memiliki prospek karir yang relevan dengan industri saat ini.",
            'source': 'Sistem Pakar Internal (Fallback)',
            'url': '#',
            'timestamp': datetime.now().isoformat()
        }

    def get_info(self, major_name):
        if major_name in self.data:
            info = self.data[major_name]
            info['source'] = 'Database Lokal (Saved)'
            return info
        
        print(f"🤖 Robot: Mempelajari jurusan {major_name} dari internet...")
        info = self._scrape_wikipedia(major_name)
        self.data[major_name] = info
        self._save_db()
        return info

    def start_background_updater(self):
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self._updater_loop, daemon=True)
            thread.start()

    def _updater_loop(self):
        majors_to_check = [
            'Teknik Informatika', 'Sistem Informasi', 'Kedokteran', 'Manajemen', 
            'Akuntansi', 'Psikologi', 'Hukum', 'Teknik Sipil'
        ]
        while self.is_running:
            for major in majors_to_check:
                if major not in self.data:
                    self.get_info(major)
                    time.sleep(5)
            time.sleep(3600)

@st.cache_resource
def get_knowledge_base():
    kb = AutonomousKnowledgeBase()
    kb.start_background_updater()
    return kb

# ==================== 3. AI ENGINE (VECTOR SPACE MODEL) ====================

class AdvancedCareerAI:
    def __init__(self):
        # Dimensi: [Logika, Verbal, Sosial, Seni, Sains]
        self.major_vectors = {
            'Teknik Informatika':     [0.9, 0.4, 0.2, 0.5, 0.6],
            'Sistem Informasi':       [0.7, 0.5, 0.6, 0.4, 0.4],
            'Kedokteran':             [0.8, 0.6, 0.7, 0.2, 1.0],
            'Psikologi':              [0.5, 0.7, 0.9, 0.3, 0.6],
            'Manajemen':              [0.6, 0.7, 0.9, 0.3, 0.2],
            'Akuntansi':              [0.9, 0.4, 0.3, 0.1, 0.2],
            'Ilmu Komunikasi':        [0.2, 0.9, 0.9, 0.6, 0.1],
            'Desain Kom. Visual':     [0.3, 0.5, 0.4, 1.0, 0.2],
            'Sastra Inggris':         [0.2, 1.0, 0.5, 0.6, 0.1],
            'Teknik Sipil':           [0.9, 0.3, 0.4, 0.5, 0.8],
            'Hukum':                  [0.6, 0.9, 0.7, 0.2, 0.2],
            'Farmasi':                [0.7, 0.4, 0.5, 0.1, 1.0],
            'Hubungan Internasional': [0.4, 0.9, 0.8, 0.2, 0.2],
            'Arsitektur':             [0.8, 0.3, 0.4, 0.9, 0.6],
            'Sastra Indonesia':       [0.2, 0.9, 0.6, 0.7, 0.1]
        }

    def _normalize_score(self, val, max_val=100):
        if val is None: return 0
        return min(max(val / max_val, 0), 1)

    def construct_user_vector(self, user_data):
        academics = user_data.get('academic_scores', {})
        interests = user_data.get('interests', {})
        competencies = user_data.get('competencies', {})
        
        # Logika Akademik
        math_wajib = academics.get('Matematika (Wajib)', 0)
        indo_wajib = academics.get('Bahasa Indonesia', 0)
        inggris_wajib = academics.get('Bahasa Inggris', 0)
        
        # Fallback Values (jika mapel tidak ada di jurusan user, nilainya 0)
        math_minat = academics.get('Matematika (Peminatan)', math_wajib)
        fisika = academics.get('Fisika', 0)
        kimia = academics.get('Kimia', 0)
        biologi = academics.get('Biologi', 0)
        ekonomi = academics.get('Ekonomi', 0)
        sosiologi = academics.get('Sosiologi', 0)
        geografi = academics.get('Geografi', 0)
        sastra_indo = academics.get('Sastra Indonesia', 0)
        sastra_inggris = academics.get('Sastra Inggris', 0)
        
        # 1. LOGIKA / MATH
        score_logika = (math_wajib * 0.3) + (math_minat * 0.3) + (fisika * 0.2) + (ekonomi * 0.2)
        vec_math = self._normalize_score(score_logika)
        
        # 2. VERBAL / LINGUAL
        avg_bahasa = (indo_wajib + inggris_wajib + sastra_indo + sastra_inggris) / 4.0
        if avg_bahasa == 0: avg_bahasa = (indo_wajib + inggris_wajib) / 2.0
        vec_verbal = (self._normalize_score(avg_bahasa) * 0.6) + \
                     (self._normalize_score(competencies.get('Komunikasi', 0), 5) * 0.4)
        
        # 3. SOSIAL / PEOPLE
        score_sosial_akad = max(sosiologi, geografi, (indo_wajib * 0.6))
        vec_social = (self._normalize_score(score_sosial_akad) * 0.3) + \
                     (self._normalize_score(interests.get('Komunikasi dan Sosial', 0), 5) * 0.4) + \
                     (self._normalize_score(competencies.get('Kepemimpinan', 0), 5) * 0.3)
        
        # 4. SENI / CREATIVE
        vec_art = (self._normalize_score(interests.get('Kreativitas dan Seni', 0), 5) * 0.6) + \
                  (self._normalize_score(competencies.get('Kreativitas', 0), 5) * 0.4)
        
        # 5. SAINS / NATURE
        score_sains = (fisika + kimia + biologi + (geografi * 0.3)) / 3.3
        vec_science = self._normalize_score(score_sains)
        
        return np.array([vec_math, vec_verbal, vec_social, vec_art, vec_science])

    def generate_recommendations(self, user_data):
        user_vector = self.construct_user_vector(user_data).reshape(1, -1)
        results = []
        
        for major, vector in self.major_vectors.items():
            major_vec = np.array(vector).reshape(1, -1)
            similarity = cosine_similarity(user_vector, major_vec)[0][0]
            match_score = similarity * 100
            
            if user_data.get('essay_analysis'):
                bonus = self._calculate_essay_bonus(user_data['essay_analysis'], major)
                match_score += bonus

            results.append({
                'major': major,
                'score': round(min(match_score, 99.9), 1),
                'vector': vector, 
                'user_vector': user_vector[0].tolist()
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

    def _calculate_essay_bonus(self, essay_analysis, major):
        bonus = 0
        text = " ".join(essay_analysis.get('overall', {}).get('key_phrases', [])).lower()
        keywords = {
            'Teknik': ['komputer', 'coding', 'mesin', 'alat'],
            'Kedokteran': ['sehat', 'dokter', 'biologi', 'bantu'],
            'Seni': ['gambar', 'desain', 'lukis', 'musik'],
            'Ekonomi': ['bisnis', 'uang', 'jual', 'usaha']
        }
        for key, words in keywords.items():
            if key in major:
                for w in words:
                    if w in text: bonus += 2
        return min(bonus, 5)

class EssayAnalyzer:
    def analyze_essays(self, essays):
        full_text = " ".join(essays.values())
        blob = TextBlob(full_text)
        polarity = blob.sentiment.polarity
        sent_label = "POSITIF" if polarity > 0.1 else "NEGATIF" if polarity < -0.1 else "NETRAL"
        return {
            'overall': {
                'sentiment': {'score': polarity, 'label': sent_label},
                'key_phrases': list(set(blob.noun_phrases))[:8],
                'word_count': len(full_text.split())
            }
        }

# ==================== 4. PDF REPORT GENERATOR ====================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'LAPORAN REKOMENDASI KARIR AI', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

# ==================== 5. USER INTERFACE (STREAMLIT) ====================

def render_step_1():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Langkah 1: Identitas Diri")
    with st.form("step1"):
        name = st.text_input("Nama Lengkap")
        stream = st.selectbox("Peminatan / Jurusan Sekolah", 
                             ["MIPA (IPA)", "IPS", "Bahasa"])
        
        if st.form_submit_button("Lanjut ➡️"):
            if not name:
                st.error("Mohon isi nama lengkap.")
            else:
                st.session_state.user_data.update({'name': name, 'stream': stream})
                st.session_state.current_step = 1
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_2():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Langkah 2: Data Akademik (Rapor SMA)")
    st.markdown('<div class="info-box">Masukkan rata-rata nilai rapor (0-100).</div>', unsafe_allow_html=True)
    st.write("")
    
    stream = st.session_state.user_data.get('stream', 'MIPA (IPA)')
    
    with st.form("step2"):
        st.markdown("#### A. Mata Pelajaran Umum (Wajib)")
        c1, c2, c3 = st.columns(3)
        with c1: math_w = st.number_input("Matematika (Wajib)", 0, 100, 75)
        with c2: indo = st.number_input("Bahasa Indonesia", 0, 100, 75)
        with c3: inggris = st.number_input("Bahasa Inggris", 0, 100, 75)
        
        st.markdown(f"#### B. Mata Pelajaran Peminatan ({stream})")
        scores_minat = {}
        
        # LOGIKA INPUT NILAI BERDASARKAN JURUSAN SMA
        if 'MIPA' in stream or 'IPA' in stream:
            c1, c2 = st.columns(2)
            with c1:
                scores_minat['Matematika (Peminatan)'] = st.number_input("Matematika Peminatan", 0, 100, 75)
                scores_minat['Fisika'] = st.number_input("Fisika", 0, 100, 75)
            with c2:
                scores_minat['Kimia'] = st.number_input("Kimia", 0, 100, 75)
                scores_minat['Biologi'] = st.number_input("Biologi", 0, 100, 75)
                
        elif 'IPS' in stream:
            c1, c2 = st.columns(2)
            with c1:
                scores_minat['Ekonomi'] = st.number_input("Ekonomi", 0, 100, 75)
                scores_minat['Sosiologi'] = st.number_input("Sosiologi", 0, 100, 75)
            with c2:
                scores_minat['Geografi'] = st.number_input("Geografi", 0, 100, 75)
                scores_minat['Sejarah Peminatan'] = st.number_input("Sejarah Peminatan", 0, 100, 75)
        
        elif 'Bahasa' in stream:
            c1, c2 = st.columns(2)
            with c1:
                scores_minat['Sastra Indonesia'] = st.number_input("Sastra Indonesia", 0, 100, 75)
                scores_minat['Sastra Inggris'] = st.number_input("Sastra Inggris", 0, 100, 75)
            with c2:
                scores_minat['Bahasa Asing Lain'] = st.number_input("Bahasa Asing (Jepang/Mandarin/dll)", 0, 100, 75)
                scores_minat['Antropologi'] = st.number_input("Antropologi", 0, 100, 75)

        if st.form_submit_button("Simpan & Lanjut ➡️"):
            all_scores = {
                'Matematika (Wajib)': math_w, 'Bahasa Indonesia': indo, 'Bahasa Inggris': inggris,
                **scores_minat
            }
            st.session_state.user_data['academic_scores'] = all_scores
            st.session_state.current_step = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_3():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Langkah 3: Minat & Kompetensi")
    with st.form("step3"):
        st.write("**Seberapa suka Anda dengan aktivitas ini? (1-5)**")
        c1, c2 = st.columns(2)
        with c1:
            i1 = st.slider("Analisis / Logika / Matematika", 1, 5, 3)
            i2 = st.slider("Sosial / Berdiskusi / Membantu Orang", 1, 5, 3)
        with c2:
            i3 = st.slider("Seni / Desain / Musik / Kreatif", 1, 5, 3)
            i4 = st.slider("Teknologi / Gadget / Coding", 1, 5, 3)
            
        st.markdown("---")
        st.write("**Penilaian Diri (Skill):**")
        k1 = st.slider("Kemampuan Komunikasi", 1, 5, 3)
        k2 = st.slider("Kreativitas", 1, 5, 3)
        k3 = st.slider("Jiwa Kepemimpinan", 1, 5, 3)

        if st.form_submit_button("Lanjut ➡️"):
            st.session_state.user_data['interests'] = {
                "Analisis dan Problem Solving": i1, "Komunikasi dan Sosial": i2,
                "Kreativitas dan Seni": i3, "Teknologi dan Programming": i4
            }
            st.session_state.user_data['competencies'] = {
                "Komunikasi": k1, "Kreativitas": k2, "Kepemimpinan": k3
            }
            st.session_state.current_step = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_4():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Langkah 4: Esai Reflektif AI")
    st.markdown('<div class="info-box">Ceritakan sedikit tentang cita-cita Anda, atau masalah apa yang ingin Anda selesaikan di dunia ini?</div>', unsafe_allow_html=True)
    st.write("")
    
    with st.form("step4"):
        essay = st.text_area("Jawaban Anda (Min. 10 kata):", height=150)
        
        if st.form_submit_button("🔍 Analisis & Lihat Hasil"):
            if len(essay.split()) < 5:
                st.error("Mohon tulis lebih panjang agar analisis akurat.")
            else:
                analyzer = EssayAnalyzer()
                analysis = analyzer.analyze_essays({'q1': essay})
                st.session_state.user_data['essay_analysis'] = analysis
                st.session_state.current_step = 4
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_results_dashboard():
    st.markdown("## 🏆 Hasil Rekomendasi Karir")
    
    ai_engine = AdvancedCareerAI()
    recommendations = ai_engine.generate_recommendations(st.session_state.user_data)
    top_rec = recommendations[0]
    kb = get_knowledge_base()
    
    # Radar Chart
    categories = ['Logika/Math', 'Verbal', 'Sosial', 'Seni', 'Sains']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=top_rec['user_vector'], theta=categories, fill='toself', 
        name='Profil Kamu', line_color='#1f77b4'
    ))
    fig.add_trace(go.Scatterpolar(
        r=top_rec['vector'], theta=categories, fill='toself', 
        name=f"Profil {top_rec['major']}", line_color='#ff7f0e', opacity=0.5
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=400)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.success(f"### 🥇 Rekomendasi: {top_rec['major']}")
        st.metric("Tingkat Kecocokan", f"{top_rec['score']}%")
        
        with st.spinner("Mengambil data pengetahuan..."):
            info = kb.get_info(top_rec['major'])
            st.markdown(f"<div class='success-box'><b>Info Jurusan:</b><br>{info['description']}</div>", unsafe_allow_html=True)
            st.caption(f"Sumber Data: {info.get('source', 'Unknown')}")
            
    st.write("### 🥈 Pilihan Alternatif")
    col_a, col_b = st.columns(2)
    for i, rec in enumerate(recommendations[1:3]):
        with (col_a if i==0 else col_b):
            with st.expander(f"{i+2}. {rec['major']} ({rec['score']}%)"):
                info_alt = kb.get_info(rec['major'])
                st.write(info_alt['description'])

    if st.button("📄 Download Laporan Lengkap (PDF)"):
        try:
            pdf = PDFReport()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Nama: {st.session_state.user_data.get('name')}", ln=1)
            pdf.cell(0, 10, f"Rekomendasi Utama: {top_rec['major']} ({top_rec['score']}%)", ln=1)
            pdf.ln(5)
            
            desc = kb.get_info(top_rec['major'])['description']
            desc_clean = desc.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, f"Deskripsi: {desc_clean}")
            
            # FIX APPLIED HERE: Removed .encode()
            pdf_out = pdf.output(dest='S') 
            b64 = base64.b64encode(pdf_out).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Hasil_Konsultasi.pdf">Klik untuk Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gagal membuat PDF: {e}")

    if st.button("🔄 Reset Aplikasi"):
        st.session_state.clear()
        st.rerun()

# ==================== MAIN APP ====================

def main():
    st.markdown('<h1 class="main-header">🇮🇩 AI Career Guidance System</h1>', unsafe_allow_html=True)
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
        st.session_state.current_step = 0
    
    steps = ["Data Diri", "Akademik", "Minat", "Esai", "Hasil"]
    st.progress((st.session_state.current_step + 1) / len(steps))
    
    step = st.session_state.current_step
    if step == 0: render_step_1()
    elif step == 1: render_step_2()
    elif step == 2: render_step_3()
    elif step == 3: render_step_4()
    elif step == 4: render_results_dashboard()

if __name__ == "__main__":
    main()