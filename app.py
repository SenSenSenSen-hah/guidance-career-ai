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
import re
import shutil

# Machine Learning & Visualization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ==================== 1. KONFIGURASI SISTEM & FAIL-SAFE NLTK ====================

NLTK_READY = False

def setup_nltk_failsafe():
    global NLTK_READY
    resources = ['punkt', 'averaged_perceptron_tagger', 'brown']
    try:
        for resource in resources: nltk.data.find(f'tokenizers/{resource}')
        NLTK_READY = True
    except LookupError:
        try:
            for resource in resources: nltk.download(resource, quiet=True)
            NLTK_READY = True
        except Exception:
            NLTK_READY = False
    except Exception:
        NLTK_READY = False

setup_nltk_failsafe()

st.set_page_config(
    page_title="AI Career Guidance (Ultimate Fix)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; font-weight: 800; }
    .step-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 3px solid #4e54c8; }
    .info-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; border-left: 4px solid #2196f3; font-size: 0.9em; }
    .success-box { background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50; font-size: 0.9em; }
    .reasoning-text { font-style: italic; color: #555; border-left: 3px solid #ff9800; padding-left: 10px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ==================== 2. ENGINE PENERJEMAH & INSIGHT GENERATOR ====================

class VectorGenerator:
    def __init__(self):
        self.keywords = {
            'math': ['matematika', 'hitung', 'logika', 'komputasi', 'algoritma', 'analisis data', 'angka', 'teknik', 'rekayasa', 'sistem', 'kalkulus', 'statistika'],
            'verbal': ['bahasa', 'sastra', 'tulis', 'baca', 'komunikasi', 'jurnalistik', 'pidato', 'inggris', 'terjemah', 'naskah', 'sejarah', 'diplomasi'],
            'social': ['sosial', 'masyarakat', 'manusia', 'bantu', 'hukum', 'politik', 'psikologi', 'manajemen', 'bisnis', 'tim', 'pimpin', 'ekonomi', 'administrasi'],
            'art': ['seni', 'desain', 'gambar', 'musik', 'kreatif', 'film', 'visual', 'estetika', 'budaya', 'arsitektur'],
            'science': ['fisika', 'kimia', 'biologi', 'alam', 'eksperimen', 'lab', 'medis', 'dokter', 'farmasi', 'lingkungan', 'kesehatan', 'reaksi', 'atom', 'gizi']
        }

    def generate_vector(self, text, major_name):
        text = text.lower()
        name = major_name.lower()
        scores = {'math': 0, 'verbal': 0, 'social': 0, 'art': 0, 'science': 0}
        
        for dim, keys in self.keywords.items():
            for key in keys:
                if key in text: scores[dim] += 0.15
        
        if 'teknik' in name: scores['math'] += 0.5
        if 'sistem' in name or 'komputer' in name: scores['math'] += 0.4
        if 'kimia' in name: 
            scores['science'] += 0.6; scores['math'] += 0.2
        if 'sastra' in name or 'bahasa' in name or 'komunikasi' in name: scores['verbal'] += 0.5
        if 'sosial' in name or 'hukum' in name or 'manajemen' in name or 'akuntansi' in name: scores['social'] += 0.4
        if 'desain' in name or 'seni' in name or 'arsitektur' in name: scores['art'] += 0.5
        if 'kedokteran' in name or 'kesehatan' in name or 'biologi' in name or 'farmasi' in name: scores['science'] += 0.5
        if 'sejarah' in name: scores['verbal'] += 0.4; scores['social'] += 0.3
            
        vector = []
        for dim in ['math', 'verbal', 'social', 'art', 'science']:
            val = min(scores[dim], 1.0); val = max(val, 0.1)
            vector.append(round(val, 2))
        return vector

class CareerInsightGenerator:
    def generate_insights(self, major_name):
        name = major_name.lower()
        
        # Default
        careers = ["Praktisi Profesional", "Konsultan", "Akademisi/Peneliti", "Wirausahawan", "Pegawai Negeri Sipil"]
        develop = ["Soft Skill Komunikasi", "Manajemen Waktu", "Bahasa Asing (Inggris)", "Literasi Digital"]
        
        if 'informatika' in name or 'komputer' in name or 'sistem' in name:
            careers = ["Software Engineer", "Data Scientist", "System Analyst", "IT Consultant", "Cyber Security Specialist"]
            develop = ["Logika Algoritma", "Bahasa Pemrograman (Python/Java)", "Problem Solving", "Portofolio Project"]
        elif 'dokter' in name or 'kedokteran' in name or 'kesehatan' in name or 'farmasi' in name:
            careers = ["Dokter/Tenaga Medis", "Peneliti Kesehatan", "Konsultan Medis", "Manajemen Rumah Sakit", "Apoteker"]
            develop = ["Ketelitian Tinggi", "Empati & Komunikasi Pasien", "Kecermatan Analisis", "Etika Profesi"]
        elif 'ekonomi' in name or 'manajemen' in name or 'akuntansi' in name or 'bisnis' in name:
            careers = ["Manajer Bisnis", "Akuntan Publik", "Financial Analyst", "Entrepreneur", "Marketing Specialist"]
            develop = ["Analisis Data Keuangan", "Kepemimpinan (Leadership)", "Negosiasi", "Penguasaan Microsoft Excel"]
        elif 'komunikasi' in name or 'hubungan' in name or 'sastra' in name:
            careers = ["Public Relations", "Content Writer/Creator", "Jurnalis", "Diplomat", "Translator"]
            develop = ["Public Speaking", "Penulisan Kreatif", "Berpikir Kritis", "Networking/Relasi"]
        elif 'desain' in name or 'seni' in name or 'arsitektur' in name:
            careers = ["Graphic Designer", "Art Director", "Arsitek", "UI/UX Designer", "Illustrator"]
            develop = ["Kreativitas Visual", "Penguasaan Software Desain", "Portofolio Karya", "Design Thinking"]
        elif 'hukum' in name:
            careers = ["Pengacara/Advokat", "Hakim/Jaksa", "Konsultan Hukum (In-house)", "Notaris", "HRD/Personalia"]
            develop = ["Analisis Kasus", "Public Speaking & Debat", "Penulisan Dokumen Legal", "Integritas"]
        elif 'teknik' in name:
            careers = ["Project Engineer", "Site Manager", "R&D Specialist", "Process Engineer", "Konsultan Teknik"]
            develop = ["Matematika Teknik", "Manajemen Proyek", "Analisis Sistem", "Sertifikasi Keahlian"]
        elif 'psikologi' in name:
            careers = ["HRD/Recruiter", "Psikolog Klinis", "Konselor Pendidikan", "Peneliti Perilaku", "Trainer Pengembangan Diri"]
            develop = ["Active Listening", "Empati", "Observasi Perilaku", "Analisis Data Statistik"]
        elif 'sejarah' in name:
            careers = ["Sejarawan", "Peneliti Budaya", "Kurator Museum", "Penulis/Jurnalis", "Pemandu Wisata Sejarah"]
            develop = ["Kemampuan Riset", "Berpikir Kritis", "Penulisan Naratif", "Analisis Dokumen"]

        return careers, develop

# ==================== 3. AUTONOMOUS KNOWLEDGE BASE (FIXED MISSING METHOD) ====================

class AutonomousKnowledgeBase:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.db_file = 'knowledge_base_v2.json'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.vector_gen = VectorGenerator()
        
        # Seed Data
        initial_majors = [
            'Teknik Informatika', 'Sistem Informasi', 'Kedokteran', 'Psikologi', 
            'Manajemen', 'Akuntansi', 'Ilmu Komunikasi', 'Desain Komunikasi Visual',
            'Sastra Inggris', 'Teknik Sipil', 'Hukum', 'Farmasi', 'Arsitektur',
            'Ilmu Sejarah', 'Hubungan Internasional', 'Sastra Indonesia',
            'Teknik Kimia', 'Teknik Industri', 'Teknik Mesin', 'Statistika'
        ]
        
        self.data = self._load_db()
        self.target_majors = list(set(initial_majors + list(self.data.keys())))
        self.is_running = False

    def _load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def _save_db(self):
        with self._lock:
            with open(self.db_file, 'w') as f: json.dump(self.data, f, indent=4)

    def _scrape_and_learn(self, major_name):
        try:
            query = major_name.replace(' ', '_')
            url = f"https://id.wikipedia.org/api/rest_v1/page/summary/{query}"
            response = requests.get(url, headers=self.headers, timeout=5)
            description = ""
            source = "Wikipedia"
            url_link = "#"
            
            if response.status_code == 200:
                res_data = response.json()
                description = res_data.get('extract', '')
                url_link = res_data.get('content_urls', {}).get('desktop', {}).get('page', '#')
            
            if not description:
                description = f"Jurusan {major_name} di perguruan tinggi."
                source = "Internal Fallback"

            generated_vector = self.vector_gen.generate_vector(description, major_name)
            return {'description': description, 'vector': generated_vector, 'source': source, 'url': url_link, 'timestamp': datetime.now().isoformat()}
        except Exception:
            return {'description': "Data tidak tersedia.", 'vector': [0.2]*5, 'source': 'Error', 'url': '#', 'timestamp': datetime.now().isoformat()}

    # [INI METODE YANG HILANG SEBELUMNYA - SUDAH DIKEMBALIKAN]
    def discover_new_majors(self):
        """Robot menjelajahi kategori Wikipedia untuk mencari jurusan baru"""
        discovery_url = "https://id.wikipedia.org/wiki/Kategori:Disiplin_akademik"
        new_found = []
        try:
            response = requests.get(discovery_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Selector untuk mengambil link artikel
                links = soup.select("#mw-pages a")
                
                for link in links:
                    major_name = link.text
                    # Filter: tidak duplikat dan panjang nama wajar
                    if major_name not in self.target_majors and len(major_name) < 35:
                        self.target_majors.append(major_name)
                        new_found.append(major_name)
            return new_found
        except Exception as e:
            print(f"Discovery Error: {e}")
            return []

    def get_all_vectors(self):
        vectors = {}
        for major in self.target_majors:
            if major not in self.data:
                self.data[major] = self._scrape_and_learn(major)
                self._save_db()
            vectors[major] = self.data[major]['vector']
        return vectors

    def get_info(self, major_name):
        if major_name not in self.data:
            self.data[major_name] = self._scrape_and_learn(major_name)
            self._save_db()
        return self.data[major_name]

@st.cache_resource
def get_knowledge_base():
    return AutonomousKnowledgeBase()

# ==================== 4. AI ENGINE (EXPLAINABLE) ====================

class AdvancedCareerAI:
    def __init__(self):
        kb = get_knowledge_base()
        self.major_vectors = kb.get_all_vectors()
        self.insight_gen = CareerInsightGenerator()

    def _normalize_score(self, val, max_val=100):
        if val is None: return 0
        return min(max(val / max_val, 0), 1)

    def construct_user_vector(self, user_data):
        academics = user_data.get('academic_scores', {})
        interests = user_data.get('interests', {})
        competencies = user_data.get('competencies', {})
        
        math_wajib = academics.get('Matematika (Wajib)', 0)
        indo_wajib = academics.get('Bahasa Indonesia', 0)
        inggris_wajib = academics.get('Bahasa Inggris', 0)
        
        math_minat = academics.get('Matematika (Peminatan)', math_wajib)
        fisika = academics.get('Fisika', 0); kimia = academics.get('Kimia', 0); biologi = academics.get('Biologi', 0)
        ekonomi = academics.get('Ekonomi', 0); sosiologi = academics.get('Sosiologi', 0); geografi = academics.get('Geografi', 0)
        sejarah_minat = academics.get('Sejarah Peminatan', 0); sastra_indo = academics.get('Sastra Indonesia', 0); sastra_inggris = academics.get('Sastra Inggris', 0)
        
        score_logika = (math_wajib * 0.3) + (math_minat * 0.3) + (fisika * 0.2) + (ekonomi * 0.2)
        vec_math = self._normalize_score(score_logika)
        
        avg_bahasa = (indo_wajib + inggris_wajib + sastra_indo + sastra_inggris + sejarah_minat) / 5.0
        if avg_bahasa == 0: avg_bahasa = (indo_wajib + inggris_wajib) / 2.0
        vec_verbal = (self._normalize_score(avg_bahasa) * 0.6) + (self._normalize_score(competencies.get('Komunikasi', 0), 5) * 0.4)
        
        score_sosial_akad = max(sosiologi, geografi, sejarah_minat, (indo_wajib * 0.6))
        vec_social = (self._normalize_score(score_sosial_akad) * 0.3) + (self._normalize_score(interests.get('Komunikasi dan Sosial', 0), 5) * 0.4) + (self._normalize_score(competencies.get('Kepemimpinan', 0), 5) * 0.3)
        
        vec_art = (self._normalize_score(interests.get('Kreativitas dan Seni', 0), 5) * 0.6) + (self._normalize_score(competencies.get('Kreativitas', 0), 5) * 0.4)
        
        score_sains = (fisika + kimia + biologi + (geografi * 0.3)) / 3.3
        vec_science = self._normalize_score(score_sains)
        
        return np.array([vec_math, vec_verbal, vec_social, vec_art, vec_science])

    def _generate_explanation(self, user_vec, major_vec, major_name, has_bonus):
        dims = ['Logika & Matematika', 'Verbal & Bahasa', 'Sosial & Humaniora', 'Seni & Kreativitas', 'Sains & Alam']
        strong_match = []
        for i in range(5):
            if user_vec[i] > 0.55 and major_vec[i] > 0.55: strong_match.append(dims[i])
        
        explanation = f"AI merekomendasikan {major_name} karena profil Anda menunjukkan kekuatan yang signifikan pada aspek {', '.join(strong_match)}."
        if not strong_match: explanation = f"AI merekomendasikan {major_name} karena profil Anda memiliki keseimbangan yang sesuai dengan karakteristik jurusan ini secara umum."
        if has_bonus: explanation += " Ditambah lagi, esai reflektif Anda mengandung kata kunci spesifik yang sangat relevan dengan bidang ini."
        return explanation

    def generate_recommendations(self, user_data):
        user_vector = self.construct_user_vector(user_data).reshape(1, -1)
        results = []
        for major, vector in self.major_vectors.items():
            major_vec = np.array(vector).reshape(1, -1)
            similarity = cosine_similarity(user_vector, major_vec)[0][0]
            match_score = similarity * 100
            bonus = 0; has_bonus = False
            if user_data.get('essay_analysis'):
                bonus = self._calculate_essay_bonus(user_data['essay_analysis'], major)
                match_score += bonus
                if bonus > 0: has_bonus = True
            
            explanation = self._generate_explanation(user_vector[0], vector, major, has_bonus)
            careers, develop = self.insight_gen.generate_insights(major)

            results.append({
                'major': major, 'score': round(min(match_score, 99.9), 1),
                'vector': vector, 'user_vector': user_vector[0].tolist(), 
                'explanation': explanation,
                'careers': careers, 'develop': develop
            })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

    def _calculate_essay_bonus(self, essay_analysis, major):
        bonus = 0
        text = " ".join(essay_analysis.get('overall', {}).get('key_phrases', [])).lower()
        if 'teknik' in major.lower() and ('teknologi' in text or 'komputer' in text): bonus += 3
        if 'sosial' in major.lower() and ('bantu' in text or 'masyarakat' in text): bonus += 3
        if 'seni' in major.lower() and ('gambar' in text or 'desain' in text): bonus += 3
        if 'kesehatan' in major.lower() and ('dokter' in text): bonus += 3
        if 'sejarah' in major.lower() and ('masa lalu' in text or 'budaya' in text): bonus += 3
        if 'kimia' in major.lower() and ('atom' in text or 'reaksi' in text or 'lab' in text): bonus += 3
        return bonus

class EssayAnalyzer:
    def analyze_essays(self, essays):
        full_text = " ".join(essays.values())
        if NLTK_READY:
            try:
                blob = TextBlob(full_text)
                polarity = blob.sentiment.polarity
                keywords = list(set(blob.noun_phrases))[:8]
            except:
                polarity = 0; keywords = self._manual_keyword_extraction(full_text)
        else:
            polarity = 0; keywords = self._manual_keyword_extraction(full_text)
        sent_label = "POSITIF" if polarity > 0.1 else "NEGATIF" if polarity < -0.1 else "NETRAL"
        return {'overall': {'sentiment': {'score': polarity, 'label': sent_label}, 'key_phrases': keywords, 'word_count': len(full_text.split())}}
    
    def _manual_keyword_extraction(self, text):
        words = re.findall(r'\w+', text.lower())
        stopwords = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'adalah', 'saya', 'ingin', 'suka']
        unique_words = list(set([w for w in words if w not in stopwords and len(w) > 4]))
        return unique_words[:8]

# ==================== 5. PDF REPORT (FINAL) ====================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'LAPORAN REKOMENDASI KARIR AI', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

# ==================== 6. UI COMPONENTS ====================

def render_step_1():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Langkah 1: Identitas Diri")
    with st.form("step1"):
        name = st.text_input("Nama Lengkap")
        stream = st.selectbox("Peminatan / Jurusan Sekolah", ["MIPA (IPA)", "IPS", "Bahasa"])
        if st.form_submit_button("Lanjut ➡️"):
            if not name: st.error("Mohon isi nama lengkap.")
            else:
                st.session_state.user_data.update({'name': name, 'stream': stream})
                st.session_state.current_step = 1
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_2():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Langkah 2: Data Akademik")
    stream = st.session_state.user_data.get('stream', 'MIPA (IPA)')
    with st.form("step2"):
        st.markdown("#### A. Mata Pelajaran Umum")
        c1, c2, c3 = st.columns(3)
        with c1: math_w = st.number_input("Matematika (Wajib)", 0, 100, 75)
        with c2: indo = st.number_input("Bahasa Indonesia", 0, 100, 75)
        with c3: inggris = st.number_input("Bahasa Inggris", 0, 100, 75)
        st.markdown(f"#### B. Mata Pelajaran Peminatan ({stream})")
        scores_minat = {}
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
                scores_minat['Sejarah Peminatan'] = st.number_input("Sejarah Peminatan (Lintas Minat)", 0, 100, 75)
        if st.form_submit_button("Simpan & Lanjut ➡️"):
            all_scores = {'Matematika (Wajib)': math_w, 'Bahasa Indonesia': indo, 'Bahasa Inggris': inggris, **scores_minat}
            st.session_state.user_data['academic_scores'] = all_scores
            st.session_state.current_step = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_3():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Langkah 3: Minat & Kompetensi")
    with st.form("step3"):
        c1, c2 = st.columns(2)
        with c1:
            i1 = st.slider("Analisis / Logika / Matematika", 1, 5, 3)
            i2 = st.slider("Sosial / Berdiskusi / Sejarah", 1, 5, 3)
        with c2:
            i3 = st.slider("Seni / Desain / Musik", 1, 5, 3)
            i4 = st.slider("Teknologi / Coding", 1, 5, 3)
        st.markdown("---")
        k1 = st.slider("Komunikasi", 1, 5, 3)
        k2 = st.slider("Kreativitas", 1, 5, 3)
        k3 = st.slider("Kepemimpinan", 1, 5, 3)
        if st.form_submit_button("Lanjut ➡️"):
            st.session_state.user_data['interests'] = {"Analisis dan Problem Solving": i1, "Komunikasi dan Sosial": i2, "Kreativitas dan Seni": i3, "Teknologi dan Programming": i4}
            st.session_state.user_data['competencies'] = {"Komunikasi": k1, "Kreativitas": k2, "Kepemimpinan": k3}
            st.session_state.current_step = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_step_4():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Langkah 4: Esai Reflektif AI")
    st.markdown("""
    <div class="info-box">
    <b>Panduan:</b> Ceritakan pelajaran apa yang kamu suka, masalah dunia apa yang ingin kamu bantu selesaikan, dan apa impian karirmu?
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    with st.form("step4"):
        essay = st.text_area("Jawaban Anda:", height=150, placeholder="Saya sangat suka...")
        if st.form_submit_button("🔍 Analisis & Lihat Hasil"):
            if len(essay.split()) < 5: st.error("Mohon tulis lebih panjang.")
            else:
                analyzer = EssayAnalyzer()
                analysis = analyzer.analyze_essays({'q1': essay})
                st.session_state.user_data['essay_analysis'] = analysis
                st.session_state.current_step = 4
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_results_dashboard():
    st.markdown("## 🏆 Hasil Rekomendasi Karir")
    
    with st.spinner("🤖 Robot sedang menghitung kecocokan dan membuat penjelasan..."):
        ai_engine = AdvancedCareerAI()
        recommendations = ai_engine.generate_recommendations(st.session_state.user_data)
        
    top_rec = recommendations[0]
    kb = get_knowledge_base()
    
    # Radar Chart
    categories = ['Logika/Math', 'Verbal', 'Sosial', 'Seni', 'Sains']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=top_rec['user_vector'], theta=categories, fill='toself', name='Profil Kamu', line_color='#1f77b4'))
    fig.add_trace(go.Scatterpolar(r=top_rec['vector'], theta=categories, fill='toself', name=f"Profil {top_rec['major']}", line_color='#ff7f0e', opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=400)
    
    c1, c2 = st.columns([1, 1])
    with c1: st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.success(f"### 🥇 Rekomendasi: {top_rec['major']}")
        st.metric("Tingkat Kecocokan", f"{top_rec['score']}%")
        info = kb.get_info(top_rec['major'])
        
        st.markdown(f"<div class='reasoning-text'><b>💡 Analisis AI:</b><br>{top_rec['explanation']}</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown(f"<div class='info-box'><b>Tentang Jurusan:</b><br>{info['description']}</div>", unsafe_allow_html=True)

    if st.button("📄 Download PDF Lengkap"):
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Nama: {st.session_state.user_data.get('name')}", ln=1)
        pdf.cell(0, 10, f"Rekomendasi Utama: {top_rec['major']} ({top_rec['score']}%)", ln=1)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Analisis Keputusan AI:", ln=1)
        pdf.set_font("Arial", '', 11)
        exp_clean = top_rec['explanation'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, exp_clean)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Deskripsi Jurusan:", ln=1)
        pdf.set_font("Arial", '', 11)
        desc_clean = kb.get_info(top_rec['major'])['description'].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, desc_clean)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Prospek Karir:", ln=1)
        pdf.set_font("Arial", '', 11)
        for career in top_rec['careers']:
            pdf.cell(0, 8, f"- {career.encode('latin-1', 'replace').decode('latin-1')}", ln=1)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Hal yang Bisa Dikembangkan:", ln=1)
        pdf.set_font("Arial", '', 11)
        for item in top_rec['develop']:
            pdf.cell(0, 8, f"- {item.encode('latin-1', 'replace').decode('latin-1')}", ln=1)
        
        pdf_out = pdf.output(dest='S')
        b64 = base64.b64encode(pdf_out).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="Laporan_Lengkap_AI.pdf">Klik untuk Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# ==================== MAIN APP ====================

def main():
    st.markdown('<h1 class="main-header">🇮🇩 Autonomous Career AI</h1>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("🤖 Kontrol Robot")
        kb = get_knowledge_base()
        st.metric("Total Jurusan Dipelajari", f"{len(kb.target_majors)} Jurusan")
        if st.button("🕵️ Eksplorasi Jurusan Baru"):
            with st.spinner("Robot sedang menjelajahi Wikipedia..."):
                new = kb.discover_new_majors()
                if new: st.success(f"Menemukan {len(new)} jurusan baru!"); kb.get_all_vectors()
                else: st.info("Belum ada jurusan baru.")
        st.markdown("---")

    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
        st.session_state.current_step = 0
    
    steps = ["Data", "Akademik", "Minat", "Esai", "Hasil"]
    st.progress((st.session_state.current_step + 1) / len(steps))
    step = st.session_state.current_step
    if step == 0: render_step_1()
    elif step == 1: render_step_2()
    elif step == 2: render_step_3()
    elif step == 3: render_step_4()
    elif step == 4: render_results_dashboard()

if __name__ == "__main__":
    main()