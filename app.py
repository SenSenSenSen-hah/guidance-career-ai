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
        for resource in resources:
            nltk.data.find(f'tokenizers/{resource}')
        NLTK_READY = True
    except LookupError:
        try:
            for resource in resources:
                nltk.download(resource, quiet=True)
            NLTK_READY = True
        except Exception:
            NLTK_READY = False
    except Exception:
        NLTK_READY = False

setup_nltk_failsafe()

st.set_page_config(
    page_title="AI Career Guidance (Math Fixed)", 
    page_icon="📐", 
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
    .rank-badge { background-color: #4e54c8; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 2. ENGINE PENERJEMAH & INSIGHT GENERATOR ====================

class VectorGenerator:
    def __init__(self):
        self.keywords = {
            'math': ['matematika', 'hitung', 'logika', 'komputasi', 'algoritma', 'analisis data', 'angka', 'teknik', 'rekayasa', 'sistem', 'kalkulus', 'statistika'],
            'verbal': ['bahasa', 'sastra', 'tulis', 'baca', 'komunikasi', 'jurnalistik', 'pidato', 'inggris', 'terjemah', 'naskah', 'sejarah', 'diplomasi', 'narasi', 'dokumen'],
            'social': ['sosial', 'masyarakat', 'manusia', 'bantu', 'hukum', 'politik', 'psikologi', 'manajemen', 'bisnis', 'tim', 'pimpin', 'ekonomi', 'administrasi', 'publik'],
            'art': ['seni', 'desain', 'gambar', 'musik', 'kreatif', 'film', 'visual', 'estetika', 'budaya', 'arsitektur', 'dekorasi'],
            'science': ['fisika', 'kimia', 'biologi', 'alam', 'eksperimen', 'lab', 'medis', 'dokter', 'farmasi', 'lingkungan', 'kesehatan', 'reaksi', 'atom', 'gizi', 'sel']
        }

    def generate_vector(self, text, major_name):
        text = text.lower()
        name = major_name.lower()
        scores = {'math': 0.1, 'verbal': 0.1, 'social': 0.1, 'art': 0.1, 'science': 0.1}
        
        # --- LOGIKA PEMISAH (DISCRIMINATOR LOGIC) ---
        
        # 1. KELOMPOK MATEMATIKA MURNI (Math Dominan Mutlak)
        if any(x in name for x in ['matematika', 'statistika', 'aktuaria']):
            scores['math'] = 0.99      # Sangat tinggi
            scores['science'] = 0.4    
            scores['social'] = 0.1     
            scores['art'] = 0.1
            
        # 2. KELOMPOK TEKNIK
        elif any(x in name for x in ['teknik', 'rekayasa', 'insinyur']):
            scores['math'] = 0.7
            scores['science'] = 0.6
            if 'kimia' in name: scores['science'] = 0.9; scores['math'] = 0.65
            if 'informatika' in name or 'komputer' in name: scores['math'] = 0.9; scores['science'] = 0.3

        # 3. KELOMPOK PERILAKU (Fix agar tidak dikira Math)
        elif any(x in name for x in ['perilaku', 'psikologi', 'bimbingan', 'konseling', 'sosial', 'sosiologi']):
            scores['social'] = 0.95
            scores['verbal'] = 0.6
            scores['math'] = 0.2       # CAP Math rendah
            
        # 4. SAINS ALAM
        elif any(x in name for x in ['fisika', 'kimia', 'biologi', 'astronomi']):
            scores['science'] = 0.9; scores['math'] = 0.6
            
        # 5. EKONOMI
        elif any(x in name for x in ['ekonomi', 'akuntansi', 'manajemen', 'bisnis']):
            scores['social'] = 0.6; scores['math'] = 0.6; scores['verbal'] = 0.4

        # 6. BAHASA/HUMANIORA
        elif any(x in name for x in ['sastra', 'bahasa', 'sejarah', 'filsafat', 'antropologi']):
            scores['verbal'] = 0.9; scores['social'] = 0.7; scores['math'] = 0.1

        # Keyword Boosting
        for dim, keys in self.keywords.items():
            count = 0
            for key in keys:
                if key in text: count += 1
            # Jangan tambah math jika jurusan sosial
            if dim == 'math' and scores['social'] > 0.8: continue
            scores[dim] += min(count * 0.1, 0.3)

        vector = []
        for dim in ['math', 'verbal', 'social', 'art', 'science']:
            val = min(scores[dim], 1.0); val = max(val, 0.1)
            vector.append(round(val, 2))
        return vector

class CareerInsightGenerator:
    def generate_insights(self, major_name):
        name = major_name.lower()
        careers = ["Praktisi Profesional", "Konsultan", "Akademisi/Peneliti", "Wirausahawan", "Pegawai Negeri Sipil"]
        develop = ["Soft Skill Komunikasi", "Manajemen Waktu", "Bahasa Asing (Inggris)", "Literasi Digital"]
        
        if 'matematika' in name or 'statistika' in name:
            careers = ["Data Scientist", "Aktuaris", "Kriptografer", "Quantitative Analyst", "Dosen Matematika"]
            develop = ["Pemrograman (R/Python)", "Analisis Statistik Lanjut", "Pemodelan Matematika"]
        elif 'teknik' in name:
            careers = ["Engineer", "Project Manager", "Site Supervisor", "R&D Specialist", "Konsultan Teknik"]
            develop = ["Sertifikasi Insinyur", "Manajemen Proyek", "Software CAD/Simulasi"]
        elif 'komputer' in name or 'informatika' in name:
            careers = ["Software Developer", "Data Analyst", "Cyber Security", "IT Consultant"]
            develop = ["Algoritma", "Portofolio GitHub", "Framework Terbaru"]
        elif 'sejarah' in name or 'sastra' in name:
            careers = ["Peneliti", "Penulis", "Kurator", "Jurnalis", "Arsiparis"]
            develop = ["Riset Arsip", "Penulisan Kreatif", "Bahasa Asing"]
            
        return careers, develop

# ==================== 3. AUTONOMOUS KNOWLEDGE BASE ====================

class AutonomousKnowledgeBase:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.db_file = 'knowledge_base_v2.json'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.vector_gen = VectorGenerator()
        
        # [FIX] Memastikan Matematika Murni ada di sini
        initial_majors = [
            'Matematika', 'Matematika Murni', 'Statistika', 'Aktuaria',
            'Teknik Informatika', 'Sistem Informasi', 'Kedokteran', 'Psikologi', 
            'Manajemen', 'Akuntansi', 'Ilmu Komunikasi', 'Desain Komunikasi Visual',
            'Sastra Inggris', 'Teknik Sipil', 'Hukum', 'Farmasi', 'Arsitektur',
            'Ilmu Sejarah', 'Hubungan Internasional', 'Sastra Indonesia',
            'Teknik Kimia', 'Teknik Industri', 'Teknik Mesin', 'Fisika', 'Biologi'
        ]
        self.data = self._load_db()
        self.target_majors = list(set(initial_majors + list(self.data.keys())))
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
            
            return {
                'description': description,
                'vector': generated_vector,
                'source': source,
                'url': url_link,
                'timestamp': datetime.now().isoformat()
            }
        except Exception:
            return {
                'description': "Data tidak tersedia.",
                'vector': [0.2, 0.2, 0.2, 0.2, 0.2],
                'source': 'Error',
                'url': '#',
                'timestamp': datetime.now().isoformat()
            }

    def discover_new_majors(self):
        urls = [
            "https://id.wikipedia.org/wiki/Kategori:Disiplin_akademik",
            "https://id.wikipedia.org/wiki/Kategori:Jurusan_pendidikan",
            "https://id.wikipedia.org/wiki/Kategori:Cabang_biologi",
            "https://id.wikipedia.org/wiki/Kategori:Cabang_fisika",
            "https://id.wikipedia.org/wiki/Kategori:Cabang_psikologi"
        ]
        new_found = []
        
        # [FIX] Blacklist diperketat untuk membuang judul kategori sampah
        blacklist = [
            "daftar", "kategori", "metode", "teori", "sejarah", "tokoh", 
            "buku", "portal", "halaman", "rintisan", "isme", "gerakan", 
            "ideologi", "konsep", "filsafat", "cabang", "terapan", 
            "perilaku", "eksperimental", "klinis", "kepribadian",
            "menurut", "bidang", "cendekiawan", "akademikus", "subyek", "berdasarkan"
        ]
        
        for discovery_url in urls:
            try:
                response = requests.get(discovery_url, headers=self.headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    links = soup.select("#mw-pages a, #mw-subcategories a")
                    for link in links:
                        raw_name = link.text
                        major_name = raw_name.replace("Kategori:", "").strip()
                        major_lower = major_name.lower()
                        
                        # Filter Cerdas
                        if (not any(bad in major_lower for bad in blacklist)) and \
                           (major_name not in self.target_majors) and \
                           (len(major_name) < 50) and (len(major_name) > 3):
                            
                            self.target_majors.append(major_name)
                            new_found.append(major_name)
            except Exception:
                continue
        return list(set(new_found))

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
        
        math_wajib = academics.get('Matematika (Wajib)', 0); indo_wajib = academics.get('Bahasa Indonesia', 0); inggris_wajib = academics.get('Bahasa Inggris', 0)
        math_minat = academics.get('Matematika (Peminatan)', math_wajib)
        fisika = academics.get('Fisika', 0); kimia = academics.get('Kimia', 0); biologi = academics.get('Biologi', 0)
        ekonomi = academics.get('Ekonomi', 0); sosiologi = academics.get('Sosiologi', 0); geografi = academics.get('Geografi', 0)
        sejarah_minat = academics.get('Sejarah Peminatan', 0); sastra_indo = academics.get('Sastra Indonesia', 0); sastra_inggris = academics.get('Sastra Inggris', 0)
        
        # LOGIKA MATEMATIKA USER DIPERKUAT
        score_logika = (math_wajib * 0.3) + (math_minat * 0.5) + (fisika * 0.1) + (ekonomi * 0.1)
        vec_math = self._normalize_score(score_logika)
        
        avg_bahasa = (indo_wajib + inggris_wajib + sastra_indo + sastra_inggris + (sejarah_minat * 1.5)) / 5.5
        if avg_bahasa == 0: avg_bahasa = (indo_wajib + inggris_wajib) / 2.0
        vec_verbal = (self._normalize_score(avg_bahasa) * 0.6) + (self._normalize_score(competencies.get('Komunikasi', 0), 5) * 0.4)
        
        score_sosial_akad = max(sosiologi, geografi, (sejarah_minat * 1.2), (indo_wajib * 0.6))
        vec_social = (self._normalize_score(score_sosial_akad) * 0.4) + (self._normalize_score(interests.get('Komunikasi dan Sosial', 0), 5) * 0.4) + (self._normalize_score(competencies.get('Kepemimpinan', 0), 5) * 0.2)
        
        vec_art = (self._normalize_score(interests.get('Kreativitas dan Seni', 0), 5) * 0.6) + (self._normalize_score(competencies.get('Kreativitas', 0), 5) * 0.4)
        
        score_sains = (fisika + kimia + biologi + (geografi * 0.3)) / 3.3
        vec_science = self._normalize_score(score_sains)
        
        return np.array([vec_math, vec_verbal, vec_social, vec_art, vec_science])

    def _generate_explanation(self, user_vec, major_vec, major_name, has_bonus):
        dims = ['Logika & Matematika', 'Verbal & Bahasa', 'Sosial & Humaniora', 'Seni & Kreativitas', 'Sains & Alam']
        strong_match = []
        for i in range(5):
            if user_vec[i] > 0.55 and major_vec[i] > 0.55:
                strong_match.append(dims[i])
        
        explanation = f"AI merekomendasikan {major_name} karena profil Anda sangat kuat pada aspek {', '.join(strong_match)}."
        if not strong_match:
            explanation = f"AI merekomendasikan {major_name} karena profil Anda memiliki keseimbangan yang sesuai."
        if has_bonus:
            explanation += " Esai Anda juga menunjukkan minat spesifik yang sangat relevan."
        return explanation

    def generate_recommendations(self, user_data):
        user_vector = self.construct_user_vector(user_data).reshape(1, -1)
        results = []
        for major, vector in self.major_vectors.items():
            major_vec = np.array(vector).reshape(1, -1)
            similarity = cosine_similarity(user_vector, major_vec)[0][0]
            match_score = similarity * 100
            
            bonus = 0
            has_bonus = False
            if user_data.get('essay_analysis'):
                bonus = self._calculate_essay_bonus(user_data['essay_analysis'], major)
                match_score += bonus
                if bonus > 0: has_bonus = True
            
            explanation = self._generate_explanation(user_vector[0], vector, major, has_bonus)
            careers, develop = self.insight_gen.generate_insights(major)

            results.append({'major': major, 'score': round(min(match_score, 99.9), 1), 'vector': vector, 'user_vector': user_vector[0].tolist(), 'explanation': explanation, 'careers': careers, 'develop': develop})
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

    def _calculate_essay_bonus(self, essay_analysis, major):
        bonus = 0
        text = " ".join(essay_analysis.get('overall', {}).get('key_phrases', [])).lower()
        name = major.lower()
        
        positive_words = ['suka', 'cinta', 'tertarik', 'senang', 'impian', 'hobi', 'antusias', 'ingin', 'mau']
        negative_words = ['benci', 'bosan', 'sulit', 'tidak suka', 'malas', 'lemah', 'bingung']
        
        is_positive_tone = any(w in essay_analysis.get('raw_text', '').lower() for w in positive_words)
        is_negative_tone = any(w in essay_analysis.get('raw_text', '').lower() for w in negative_words)
        
        if is_positive_tone and not is_negative_tone:
            if 'matematika' in name and ('logika' in text or 'angka' in text or 'hitung' in text):
                bonus += 7
            if ('sejarah' in name or 'arkeologi' in name) and ('sejarah' in text or 'masa lalu' in text or 'kuno' in text):
                bonus += 7
            if 'teknik' in name and ('teknologi' in text or 'mesin' in text):
                bonus += 3
            if 'kimia' in name and ('kimia' in text or 'reaksi' in text):
                bonus += 5
            if 'sosial' in name and ('masyarakat' in text):
                bonus += 3
            if 'seni' in name and ('gambar' in text):
                bonus += 3
            if 'dokter' in name and ('kesehatan' in text):
                bonus += 3
        return bonus

class EssayAnalyzer:
    def analyze_essays(self, essays):
        full_text = " ".join(essays.values())
        
        positive_words = ['suka', 'cinta', 'tertarik', 'senang', 'impian', 'hobi', 'antusias', 'ingin', 'mau']
        negative_words = ['benci', 'bosan', 'sulit', 'tidak suka', 'malas', 'lemah', 'bingung']
        
        pos_count = sum(1 for w in positive_words if w in full_text.lower())
        neg_count = sum(1 for w in negative_words if w in full_text.lower())
        
        if pos_count > neg_count: sent_label = "POSITIF (Antusias)"
        elif neg_count > pos_count: sent_label = "NEGATIF (Kurang Minat)"
        else: sent_label = "NETRAL"
        
        if NLTK_READY:
            try:
                blob = TextBlob(full_text)
                keywords = list(set(blob.noun_phrases))[:10]
            except:
                keywords = self._manual_keyword_extraction(full_text)
        else:
            keywords = self._manual_keyword_extraction(full_text)
            
        keywords.extend(self._manual_keyword_extraction(full_text))
        
        return {
            'overall': {
                'sentiment': {'label': sent_label},
                'key_phrases': list(set(keywords)),
                'word_count': len(full_text.split())
            },
            'raw_text': full_text
        }
    
    def _manual_keyword_extraction(self, text):
        words = re.findall(r'\w+', text.lower())
        stopwords = ['yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'adalah', 'saya', 'ingin', 'suka', 'sangat', 'mau', 'akan']
        return list(set([w for w in words if w not in stopwords and len(w) > 4]))

# ==================== 5. PDF REPORT (MULTI RECOMMENDATION) ====================

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
            if not name:
                st.error("Mohon isi nama lengkap.")
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
        with c1:
            math_w = st.number_input("Matematika (Wajib)", 0, 100, 75)
        with c2:
            indo = st.number_input("Bahasa Indonesia", 0, 100, 75)
        with c3:
            inggris = st.number_input("Bahasa Inggris", 0, 100, 75)
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
    st.markdown("""
    <div class="info-box">
    <b>Panduan Pengisian Skala (1-5):</b><br>
    1️⃣ = <b>Sangat Rendah / Tidak Suka</b><br>
    5️⃣ = <b>Sangat Tinggi / Sangat Suka</b>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
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
    st.markdown("""<div class="info-box"><b>Panduan:</b> Ceritakan pelajaran apa yang kamu suka, masalah dunia apa yang ingin kamu bantu selesaikan, dan apa impian karirmu?</div>""", unsafe_allow_html=True)
    st.write("")
    with st.form("step4"):
        essay = st.text_area("Jawaban Anda:", height=150, placeholder="Saya sangat suka matematika dan logika...")
        if st.form_submit_button("🔍 Analisis & Lihat Hasil"):
            if len(essay.split()) < 5:
                st.error("Mohon tulis lebih panjang.")
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
    
    top_3 = recommendations[:3]
    kb = get_knowledge_base()
    
    # Radar Chart
    categories = ['Logika/Math', 'Verbal', 'Sosial', 'Seni', 'Sains']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=top_3[0]['user_vector'], theta=categories, fill='toself', name='Profil Kamu', line_color='#1f77b4'))
    fig.add_trace(go.Scatterpolar(r=top_3[0]['vector'], theta=categories, fill='toself', name=f"Profil {top_3[0]['major']}", line_color='#ff7f0e', opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Detail Pilihan Terbaik")
    tab1, tab2, tab3 = st.tabs([f"🥇 {top_3[0]['major']}", f"🥈 {top_3[1]['major']}", f"🥉 {top_3[2]['major']}"])
    
    for i, tab in enumerate([tab1, tab2, tab3]):
        rec = top_3[i]
        info = kb.get_info(rec['major'])
        with tab:
            st.success(f"**Skor Kecocokan: {rec['score']}%**")
            st.markdown(f"<div class='reasoning-text'><b>💡 Analisis AI:</b><br>{rec['explanation']}</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"**📖 Deskripsi Jurusan:**\n{info['description']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**💼 Prospek Karir:**")
                for car in rec['careers']:
                    st.write(f"- {car}")
            with c2:
                st.markdown("**📈 Hal yang Perlu Dikembangkan:**")
                for dev in rec['develop']:
                    st.write(f"- {dev}")

    if st.button("📄 Download PDF (3 Rekomendasi)"):
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "HASIL ANALISIS KARIR AI", 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Nama Peserta: {st.session_state.user_data.get('name')}", ln=1)
        pdf.cell(0, 10, f"Tanggal: {datetime.now().strftime('%d-%m-%Y')}", ln=1)
        pdf.ln(10)
        
        for i, rec in enumerate(top_3):
            info = kb.get_info(rec['major'])
            if i > 0:
                pdf.add_page()
            
            title_rank = f"PILIHAN {i+1}: {rec['major']} ({rec['score']}%)"
            pdf.set_fill_color(230, 230, 250)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 12, title_rank, 0, 1, 'L', 1)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Analisis AI:", ln=1)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, rec['explanation'].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Deskripsi Jurusan:", ln=1)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, info['description'].encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Prospek Karir:", ln=1)
            pdf.set_font("Arial", '', 11)
            for c in rec['careers']:
                pdf.cell(0, 6, f"- {c.encode('latin-1', 'replace').decode('latin-1')}", ln=1)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, "Pengembangan Diri:", ln=1)
            pdf.set_font("Arial", '', 11)
            for d in rec['develop']:
                pdf.cell(0, 6, f"- {d.encode('latin-1', 'replace').decode('latin-1')}", ln=1)

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
        st.header("🤖 Pusat Kontrol AI")
        kb = get_knowledge_base()
        st.metric("Kapasitas Otak", f"{len(kb.target_majors)} Item")
        
        st.subheader("📡 Discovery Agent")
        if st.button("🕵️ Cari Jurusan Baru"):
            with st.spinner("Robot sedang menjelajahi Wikipedia..."):
                new = kb.discover_new_majors()
                if new:
                    st.success(f"Menemukan {len(new)} jurusan baru!")
                    kb.get_all_vectors()
                else:
                    st.info("Belum ada jurusan baru.")
        st.markdown("---")
        st.subheader("🔒 Area Admin")
        
        show_admin = st.checkbox("Kelola Memori (Admin)")
        if show_admin:
            password = st.text_input("Masukkan Password Admin:", type="password")
            if password == "admin123":
                st.success("Akses Diterima ✅")
                st.write("**💾 Database Control**")
                if os.path.exists(kb.db_file):
                    with open(kb.db_file, "rb") as f:
                        st.download_button("📥 Backup Otak (JSON)", f, "backup_knowledge_base.json")
                
                uploaded = st.file_uploader("📤 Restore Memori", type=["json"])
                if uploaded:
                    try:
                        data = json.load(uploaded)
                        with open(kb.db_file, "w") as f:
                            json.dump(data, f, indent=4)
                        st.success("✅ Memori dipulihkan!")
                        time.sleep(1)
                        st.rerun()
                    except:
                        st.error("File rusak.")
            elif password:
                st.error("Password Salah ❌")

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