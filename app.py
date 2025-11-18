import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from datetime import datetime
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from textblob import TextBlob
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="AI Career Guidance Platform",
    page_icon="🎓",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .step-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .scraping-info {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .debug-panel {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .essay-analysis {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 10px 0;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== TEXT ANALYSIS FUNCTIONS ====================
class EssayAnalyzer:
    def __init__(self):
        self.interest_keywords = {
            'teknologi': ['programming', 'coding', 'komputer', 'teknologi', 'software', 'aplikasi', 'digital', 'internet', 'website', 'ai', 'artificial intelligence'],
            'sains': ['penelitian', 'eksperimen', 'sains', 'ilmiah', 'laboratorium', 'biologi', 'kimia', 'fisika', 'matematika', 'analisis data'],
            'seni': ['seni', 'desain', 'kreatif', 'gambar', 'musik', 'menulis', 'fotografi', 'film', 'tari', 'lukis'],
            'sosial': ['masyarakat', 'sosial', 'membantu', 'komunitas', 'volunteer', 'organisasi', 'gotong royong', 'peduli'],
            'bisnis': ['bisnis', 'wirausaha', 'perusahaan', 'manajemen', 'marketing', 'jualan', 'profit', 'usaha'],
            'kesehatan': ['kesehatan', 'dokter', 'perawat', 'rumah sakit', 'medis', 'obat', 'pasien', 'penyakit'],
            'pendidikan': ['mengajar', 'guru', 'sekolah', 'belajar', 'pendidikan', 'ilmu', 'pengetahuan'],
            'teknik': ['teknik', 'mesin', 'elektro', 'bangunan', 'konstruksi', 'engineer', 'perancangan']
        }
        
        self.competency_keywords = {
            'analisis': ['menganalisis', 'meneliti', 'memecahkan', 'problem solving', 'logika', 'data'],
            'kreativitas': ['mencipta', 'ide', 'inovasi', 'kreatif', 'imajinasi', 'desain'],
            'komunikasi': ['berbicara', 'presentasi', 'menulis', 'diskusi', 'negosiasi', 'jelas'],
            'kepemimpinan': ['memimpin', 'mengorganisir', 'koordinasi', 'tim', 'proyek', 'inisiatif'],
            'ketelitian': ['detail', 'teliti', 'akurat', 'presisi', 'hati-hati', 'perfeksionis']
        }
    
    def analyze_sentiment(self, text):
        """Analisis sentimen menggunakan TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return {'score': polarity, 'label': 'POSITIF', 'emoji': '😊'}
            elif polarity < -0.1:
                return {'score': polarity, 'label': 'NEGATIF', 'emoji': '😔'}
            else:
                return {'score': polarity, 'label': 'NETRAL', 'emoji': '😐'}
        except:
            return {'score': 0, 'label': 'NETRAL', 'emoji': '😐'}
    
    def extract_interests(self, text):
        """Ekstrak minat dari teks esai"""
        text_lower = text.lower()
        found_interests = []
        
        for interest_category, keywords in self.interest_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_interests.append(interest_category)
        
        return found_interests
    
    def extract_competencies(self, text):
        """Ekstrak kompetensi dari teks esai"""
        text_lower = text.lower()
        found_competencies = []
        
        for competency_category, keywords in self.competency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_competencies.append(competency_category)
        
        return found_competencies
    
    def extract_key_phrases(self, text, num_phrases=5):
        """Ekstrak frase kunci dari teks"""
        try:
            # Simple approach: extract meaningful phrases
            sentences = nltk.sent_tokenize(text)
            words = []
            for sentence in sentences:
                words.extend(sentence.split())
            
            # Filter meaningful words (remove common words)
            common_words = {'saya', 'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'pada', 'dengan', 'ini', 'itu', 'adalah'}
            meaningful_words = [word for word in words if word.lower() not in common_words and len(word) > 3]
            
            # Count frequency
            word_freq = Counter(meaningful_words)
            return [word for word, count in word_freq.most_common(num_phrases)]
        except:
            return []
    
    def analyze_essays(self, essays):
        """Analisis komprehensif semua esai"""
        all_analysis = {}
        combined_text = ""
        
        for i, (question, answer) in enumerate(essays.items()):
            if answer.strip():
                analysis = {
                    'sentiment': self.analyze_sentiment(answer),
                    'interests': self.extract_interests(answer),
                    'competencies': self.extract_competencies(answer),
                    'key_phrases': self.extract_key_phrases(answer),
                    'word_count': len(answer.split()),
                    'answer': answer
                }
                all_analysis[f'essay_{i+1}'] = analysis
                combined_text += " " + answer
        
        # Overall analysis
        if combined_text:
            overall_analysis = {
                'sentiment': self.analyze_sentiment(combined_text),
                'interests': self.extract_interests(combined_text),
                'competencies': self.extract_competencies(combined_text),
                'key_phrases': self.extract_key_phrases(combined_text),
                'total_word_count': len(combined_text.split())
            }
            all_analysis['overall'] = overall_analysis
        
        return all_analysis

# ==================== WEB SCRAPING FUNCTIONS ====================
class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_university_info(self, major_name):
        """Scrape informasi jurusan dari berbagai sumber"""
        try:
            # Coba beberapa user agents
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ]
            
            for user_agent in user_agents:
                self.headers['User-Agent'] = user_agent
                
                # Sumber-sumber yang akan di-scrape
                sources = [
                    self._scrape_from_wikipedia(major_name),
                    self._scrape_from_zenius(major_name),
                    self._scrape_from_quipper(major_name)
                ]
                
                # Filter hasil yang valid
                valid_sources = [source for source in sources if source and source.get('description')]
                
                if valid_sources:
                    return valid_sources[0]  # Return yang pertama yang valid
            
            return self._generate_fallback_info(major_name)
                
        except Exception as e:
            return self._generate_fallback_info(major_name)
    
    def _scrape_from_wikipedia(self, major_name):
        """Scrape dari Wikipedia Indonesia"""
        try:
            url = f"https://id.wikipedia.org/wiki/{major_name.replace(' ', '_')}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Ambil paragraf pertama
                content_div = soup.find('div', {'class': 'mw-parser-output'})
                if content_div:
                    paragraphs = content_div.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 100:  # Ambil yang cukup panjang
                            return {
                                'description': text[:500] + '...' if len(text) > 500 else text,
                                'source': 'Wikipedia',
                                'url': url,
                                'status': 'success'
                            }
            return None
        except:
            return None
    
    def _scrape_from_zenius(self, major_name):
        """Scrape dari website Zenius (contoh)"""
        try:
            # Contoh scraping dari situs pendidikan
            url = f"https://www.zenius.net/blog/{major_name.replace(' ', '-').lower()}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Cari konten artikel
                article_content = soup.find('article')
                if article_content:
                    paragraphs = article_content.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 50:
                            return {
                                'description': text[:400] + '...' if len(text) > 400 else text,
                                'source': 'Zenius Education',
                                'url': url,
                                'status': 'success'
                            }
            return None
        except:
            return None
    
    def _scrape_from_quipper(self, major_name):
        """Scrape dari website Quipper (contoh)"""
        try:
            # Contoh scraping dari situs pendidikan
            url = f"https://www.quipper.com/id/blog/{major_name.replace(' ', '-').lower()}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Cari konten blog
                blog_content = soup.find('div', class_='blog-content')
                if blog_content:
                    paragraphs = blog_content.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 50:
                            return {
                                'description': text[:400] + '...' if len(text) > 400 else text,
                                'source': 'Quipper',
                                'url': url,
                                'status': 'success'
                            }
            return None
        except:
            return None
    
    def _generate_fallback_info(self, major_name):
        """Generate informasi fallback jika scraping gagal"""
        fallback_descriptions = {
            'Teknik Informatika': 'Jurusan Teknik Informatika mempelajari pengembangan perangkat lunak, artificial intelligence, dan sistem komputer. Lulusannya bekerja sebagai software engineer, data scientist, dan IT consultant. Prospek karir sangat tinggi dengan permintaan yang terus meningkat di era digital.',
            'Kedokteran': 'Jurusan Kedokteran mempelajari ilmu medis, anatomi manusia, dan praktik klinis. Lulusannya menjadi dokter umum atau spesialis di rumah sakit dan klinik. Membutuhkan komitmen tinggi dan masa studi yang panjang.',
            'Manajemen': 'Jurusan Manajemen fokus pada pengelolaan bisnis, pemasaran, dan sumber daya manusia. Lulusannya bekerja sebagai manajer, konsultan bisnis, atau entrepreneur. Cocok untuk yang memiliki jiwa kepemimpinan.',
            'Psikologi': 'Jurusan Psikologi mempelajari perilaku manusia dan proses mental. Lulusannya bekerja sebagai psikolog klinis, HR specialist, atau konselor. Membutuhkan empati dan kemampuan analisis yang baik.',
            'Sastra Inggris': 'Jurusan Sastra Inggris mempelajari bahasa, sastra, dan budaya Inggris. Lulusannya bekerja sebagai penerjemah, content writer, atau diplomat. Peluang karir di bidang pendidikan dan media juga terbuka lebar.',
            'Teknik Elektro': 'Jurusan Teknik Elektro mempelajari sistem kelistrikan, elektronika, dan telekomunikasi. Lulusannya bekerja di bidang energi, telekomunikasi, dan manufaktur elektronik.',
            'Farmasi': 'Jurusan Farmasi fokus pada ilmu obat-obatan dan farmakologi. Lulusannya menjadi apoteker, peneliti obat, atau bekerja di industri farmasi.',
            'Akuntansi': 'Jurusan Akuntansi mempelajari pencatatan dan analisis keuangan. Lulusannya menjadi akuntan, auditor, atau konsultan pajak.',
            'Ilmu Komunikasi': 'Jurusan Ilmu Komunikasi mempelajari media, public relations, dan jurnalistik. Lulusannya bekerja di media, advertising, atau corporate communication.',
            'Sastra Indonesia': 'Jurusan Sastra Indonesia mempelajari bahasa dan sastra Indonesia. Lulusannya menjadi penulis, editor, atau bekerja di bidang budaya dan pendidikan.',
            'Penerjemahan': 'Jurusan Penerjemahan fokus pada teknik penerjemahan antar bahasa. Lulusannya menjadi penerjemah, interpreter, atau language specialist.'
        }
        
        description = fallback_descriptions.get(major_name, 
            f"Jurusan {major_name} merupakan program studi yang relevan dengan minat dan bakat Anda. Disarankan untuk mencari informasi lebih lanjut dari sumber terpercaya mengenai kurikulum dan prospek karirnya.")
        
        return {
            'description': description,
            'source': 'Sistem AI Career Guidance',
            'url': '#',
            'status': 'fallback'
        }

# ==================== PDF REPORT GENERATOR ====================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'LAPORAN KONSULTASI KARIR', 0, 1, 'C')
        self.set_font('Arial', 'I', 12)
        self.cell(0, 10, 'AI Career Guidance Platform', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')
    
    def add_section_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
    
    def add_recommendation_card(self, rank, major, score, reasoning, careers, scraped_info=None):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, f'{rank}. {major} - Skor: {score}%', 0, 1)
        
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, f'Alasan: {reasoning}')
        
        # Tambahkan informasi dari web scraping jika ada
        if scraped_info and scraped_info.get('description'):
            self.set_font('Arial', 'I', 9)
            self.multi_cell(0, 6, f"Info: {scraped_info['description']}")
            self.cell(0, 6, f"Sumber: {scraped_info.get('source', 'Unknown')}", 0, 1)
        
        self.set_font('Arial', 'I', 10)
        self.cell(0, 6, 'Peluang Karir:', 0, 1)
        for career in careers:
            self.cell(10)  # indent
            self.cell(0, 6, f'• {career}', 0, 1)
        
        self.ln(5)

def generate_pdf_report(user_data, recommendations, scraped_data, essay_analysis=None):
    pdf = PDFReport()
    pdf.add_page()
    
    # Header
    pdf.add_section_title('📋 Data Peserta')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Nama: {user_data.get("name", "N/A")}', 0, 1)
    pdf.cell(0, 8, f'Peminatan: {user_data.get("stream", "N/A")}', 0, 1)
    pdf.cell(0, 8, f'Tanggal Konsultasi: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1)
    
    # Academic Scores (if available)
    if user_data.get('academic_scores'):
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Nilai Akademik:', 0, 1)
        pdf.set_font('Arial', '', 10)
        for subject, score in user_data['academic_scores'].items():
            pdf.cell(20)  # indent
            pdf.cell(0, 6, f'{subject}: {score}', 0, 1)
    
    # Essay Analysis (if available)
    if essay_analysis and essay_analysis.get('overall'):
        pdf.ln(5)
        pdf.add_section_title('📝 Analisis Esai Reflektif')
        pdf.set_font('Arial', '', 10)
        overall = essay_analysis['overall']
        
        pdf.cell(0, 6, f'Sentimen: {overall.get("sentiment", {}).get("label", "N/A")}', 0, 1)
        
        if overall.get('interests'):
            pdf.cell(0, 6, f'Minat Terdeteksi: {", ".join(overall["interests"])}', 0, 1)
        
        if overall.get('competencies'):
            pdf.cell(0, 6, f'Kompetensi Terdeteksi: {", ".join(overall["competencies"])}', 0, 1)
        
        if overall.get('key_phrases'):
            pdf.cell(0, 6, f'Kata Kunci: {", ".join(overall["key_phrases"][:5])}', 0, 1)
    
    # Recommendations dengan data scraping
    pdf.add_section_title('🎓 Rekomendasi Jurusan')
    for i, rec in enumerate(recommendations, 1):
        scraped_info = scraped_data.get(rec['major'])
        pdf.add_recommendation_card(
            i, rec['major'], rec['score'], 
            rec['reasoning'], rec['careers'],
            scraped_info
        )
    
    # Career Plan
    pdf.add_section_title('📅 Rencana Pengembangan Karir')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Jangka Pendek (1-2 tahun):', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, '• Fokus pada mata pelajaran prasyarat\n• Ikuti ekstrakurikuler yang relevan\n• Eksplorasi melalui kursus online')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Jangka Menengah (2-4 tahun):', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, '• Persiapan masuk perguruan tinggi\n• Cari pengalaman magang atau volunteering\n• Bangun jaringan profesional')
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Jangka Panjang (4+ tahun):', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, '• Selesaikan pendidikan tinggi\n• Dapatkan sertifikasi profesional\n• Terus belajar dan beradaptasi dengan perkembangan')
    
    # Notes
    pdf.add_section_title('💡 Catatan Penting')
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 6, 'Hasil ini merupakan rekomendasi berdasarkan data yang diberikan. Disarankan untuk berkonsultasi lebih lanjut dengan konselor karir untuk keputusan yang lebih tepat.')
    
    return pdf.output(dest='S').encode('latin1')

# ==================== CAREER AI SYSTEM ====================
class CareerAI:
    def __init__(self):
        self.majors_data = {
            'IPA': [
                {'name': 'Teknik Informatika', 'skills': ['Programming', 'Matematika', 'Logika'], 'prospects': 'Sangat Tinggi'},
                {'name': 'Kedokteran', 'skills': ['Biologi', 'Kimia', 'Analisis'], 'prospects': 'Tinggi'},
                {'name': 'Teknik Elektro', 'skills': ['Fisika', 'Matematika', 'Problem Solving'], 'prospects': 'Tinggi'},
                {'name': 'Farmasi', 'skills': ['Kimia', 'Biologi', 'Ketelitian'], 'prospects': 'Tinggi'}
            ],
            'IPS': [
                {'name': 'Manajemen', 'skills': ['Komunikasi', 'Analisis', 'Kepemimpinan'], 'prospects': 'Tinggi'},
                {'name': 'Akuntansi', 'skills': ['Matematika', 'Ketelitian', 'Analisis'], 'prospects': 'Tinggi'},
                {'name': 'Ilmu Komunikasi', 'skills': ['Kreativitas', 'Komunikasi', 'Writing'], 'prospects': 'Sedang'},
                {'name': 'Psikologi', 'skills': ['Empati', 'Analisis', 'Komunikasi'], 'prospects': 'Tinggi'}
            ],
            'Bahasa': [
                {'name': 'Sastra Inggris', 'skills': ['Bahasa', 'Analisis', 'Writing'], 'prospects': 'Sedang'},
                {'name': 'Sastra Indonesia', 'skills': ['Bahasa', 'Kreativitas', 'Analisis'], 'prospects': 'Sedang'},
                {'name': 'Penerjemahan', 'skills': ['Bahasa', 'Ketelitian', 'Komunikasi'], 'prospects': 'Tinggi'}
            ]
        }
        
        self.career_paths = {
            'Teknik Informatika': ['Software Engineer', 'Data Scientist', 'Web Developer', 'AI Specialist'],
            'Kedokteran': ['Dokter Umum', 'Dokter Spesialis', 'Peneliti Medis', 'Dosen Kedokteran'],
            'Teknik Elektro': ['Electrical Engineer', 'Telecommunication Engineer', 'Power Systems Engineer', 'Electronics Designer'],
            'Farmasi': ['Apoteker', 'Peneliti Obat', 'Medical Representative', 'Quality Control'],
            'Manajemen': ['Manajer Perusahaan', 'Entrepreneur', 'Konsultan Bisnis', 'Business Analyst'],
            'Akuntansi': ['Akuntan', 'Auditor', 'Konsultan Pajak', 'Financial Analyst'],
            'Ilmu Komunikasi': ['Jurnalis', 'Public Relations', 'Content Creator', 'Media Planner'],
            'Psikologi': ['Psikolog Klinis', 'HR Specialist', 'Konselor', 'Researcher'],
            'Sastra Inggris': ['Penerjemah', 'Content Writer', 'Diplomat', 'Teacher'],
            'Sastra Indonesia': ['Penulis', 'Editor', 'Jurnalis', 'Content Creator'],
            'Penerjemahan': ['Translator', 'Interpreter', 'Language Specialist', 'Localization Expert']
        }
        
        self.scraper = WebScraper()
        self.essay_analyzer = EssayAnalyzer()
    
    def calculate_match_score(self, user_data, major):
        """Calculate match score dengan algoritma lebih sophisticated"""
        score = 50  # Base score lebih rendah
        
        # 1. Academic Match (30%)
        academic_score = 0
        if user_data.get('academic_scores'):
            subject_weights = self.get_subject_weights(major['name'])
            total_weight = 0
            for subject, weight in subject_weights.items():
                if subject in user_data['academic_scores']:
                    academic_score += (user_data['academic_scores'][subject] / 100) * weight
                    total_weight += weight
            if total_weight > 0:
                academic_score = (academic_score / total_weight) * 100
            score += academic_score * 0.30  # 30%
        
        # 2. Interest Match (25%)
        interest_score = 0
        if user_data.get('interests'):
            interest_weights = self.get_interest_weights(major['name'])
            total_interest_weight = 0
            for interest, rating in user_data['interests'].items():
                weight = interest_weights.get(interest, 0.5)
                interest_score += (rating / 5) * weight
                total_interest_weight += weight
            if total_interest_weight > 0:
                interest_score = (interest_score / total_interest_weight) * 100
            score += interest_score * 0.25  # 25%
        
        # 3. Competency Match (20%)
        competency_score = 0
        if user_data.get('competencies'):
            competency_weights = self.get_competency_weights(major['name'])
            total_competency_weight = 0
            for competency, rating in user_data['competencies'].items():
                weight = competency_weights.get(competency, 0.5)
                competency_score += (rating / 5) * weight
                total_competency_weight += weight
            if total_competency_weight > 0:
                competency_score = (competency_score / total_competency_weight) * 100
            score += competency_score * 0.20  # 20%
        
        # 4. Essay Analysis Match (25%)
        essay_score = 0
        if user_data.get('essay_analysis') and user_data['essay_analysis'].get('overall'):
            overall = user_data['essay_analysis']['overall']
            
            # Match interests from essays
            essay_interests = set(overall.get('interests', []))
            major_interests = set(self.get_major_interests(major['name']))
            if essay_interests and major_interests:
                interest_match = len(essay_interests.intersection(major_interests)) / len(major_interests)
                essay_score += interest_match * 50
            
            # Match competencies from essays
            essay_competencies = set(overall.get('competencies', []))
            major_competencies = set(self.get_major_competencies(major['name']))
            if essay_competencies and major_competencies:
                competency_match = len(essay_competencies.intersection(major_competencies)) / len(major_competencies)
                essay_score += competency_match * 30
            
            # Sentiment bonus
            sentiment = overall.get('sentiment', {}).get('score', 0)
            essay_score += max(sentiment * 20, 0)  # Positive sentiment gives bonus
            
            score += essay_score * 0.25  # 25%
        
        # 5. Random variation untuk menghindari hasil identik
        score += random.uniform(-3, 3)
        
        return min(max(score, 0), 100)
    
    def get_subject_weights(self, major_name):
        """Return subject weights berdasarkan jurusan"""
        weights = {
            'Teknik Informatika': {'Matematika': 0.4, 'Fisika': 0.3, 'Kimia': 0.1, 'Biologi': 0.0},
            'Kedokteran': {'Matematika': 0.2, 'Fisika': 0.2, 'Kimia': 0.3, 'Biologi': 0.3},
            'Teknik Elektro': {'Matematika': 0.4, 'Fisika': 0.4, 'Kimia': 0.1, 'Biologi': 0.0},
            'Farmasi': {'Matematika': 0.2, 'Fisika': 0.1, 'Kimia': 0.4, 'Biologi': 0.3},
            'Manajemen': {'Matematika': 0.3, 'Ekonomi': 0.4, 'Sejarah': 0.1, 'Geografi': 0.1},
            'Akuntansi': {'Matematika': 0.4, 'Ekonomi': 0.3, 'Sejarah': 0.1, 'Geografi': 0.1},
            'Ilmu Komunikasi': {'Matematika': 0.1, 'Ekonomi': 0.2, 'Sejarah': 0.3, 'Geografi': 0.2},
            'Psikologi': {'Matematika': 0.1, 'Ekonomi': 0.2, 'Sejarah': 0.3, 'Geografi': 0.2},
            'Sastra Inggris': {'Bahasa Indonesia': 0.3, 'Bahasa Inggris': 0.4, 'Sastra': 0.2, 'Sejarah': 0.1},
            'Sastra Indonesia': {'Bahasa Indonesia': 0.4, 'Bahasa Inggris': 0.2, 'Sastra': 0.3, 'Sejarah': 0.1},
            'Penerjemahan': {'Bahasa Indonesia': 0.3, 'Bahasa Inggris': 0.4, 'Sastra': 0.2, 'Sejarah': 0.1}
        }
        return weights.get(major_name, {'Matematika': 0.25, 'Fisika': 0.25, 'Kimia': 0.25, 'Biologi': 0.25})
    
    def get_interest_weights(self, major_name):
        """Return interest weights berdasarkan jurusan"""
        weights = {
            'Teknik Informatika': {'Analisis dan Problem Solving': 0.4, 'Teknologi dan Programming': 0.4, 'Kreativitas dan Seni': 0.1, 'Komunikasi dan Sosial': 0.1},
            'Kedokteran': {'Analisis dan Problem Solving': 0.3, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.1, 'Komunikasi dan Sosial': 0.5},
            'Teknik Elektro': {'Analisis dan Problem Solving': 0.4, 'Teknologi dan Programming': 0.3, 'Kreativitas dan Seni': 0.1, 'Komunikasi dan Sosial': 0.2},
            'Farmasi': {'Analisis dan Problem Solving': 0.3, 'Teknologi dan Programming': 0.2, 'Kreativitas dan Seni': 0.1, 'Komunikasi dan Sosial': 0.4},
            'Manajemen': {'Analisis dan Problem Solving': 0.3, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.2, 'Komunikasi dan Sosial': 0.4},
            'Akuntansi': {'Analisis dan Problem Solving': 0.4, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.1, 'Komunikasi dan Sosial': 0.4},
            'Ilmu Komunikasi': {'Analisis dan Problem Solving': 0.2, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.3, 'Komunikasi dan Sosial': 0.4},
            'Psikologi': {'Analisis dan Problem Solving': 0.2, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.2, 'Komunikasi dan Sosial': 0.5},
            'Sastra Inggris': {'Analisis dan Problem Solving': 0.2, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.4, 'Komunikasi dan Sosial': 0.3},
            'Sastra Indonesia': {'Analisis dan Problem Solving': 0.2, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.4, 'Komunikasi dan Sosial': 0.3},
            'Penerjemahan': {'Analisis dan Problem Solving': 0.3, 'Teknologi dan Programming': 0.1, 'Kreativitas dan Seni': 0.2, 'Komunikasi dan Sosial': 0.4}
        }
        return weights.get(major_name, {'Analisis dan Problem Solving': 0.25, 'Teknologi dan Programming': 0.25, 'Kreativitas dan Seni': 0.25, 'Komunikasi dan Sosial': 0.25})
    
    def get_competency_weights(self, major_name):
        """Return competency weights berdasarkan jurusan"""
        weights = {
            'Teknik Informatika': {'Analisis Logis': 0.4, 'Kreativitas': 0.3, 'Komunikasi': 0.1, 'Kepemimpinan': 0.2},
            'Kedokteran': {'Analisis Logis': 0.3, 'Kreativitas': 0.1, 'Komunikasi': 0.4, 'Kepemimpinan': 0.2},
            'Teknik Elektro': {'Analisis Logis': 0.4, 'Kreativitas': 0.2, 'Komunikasi': 0.2, 'Kepemimpinan': 0.2},
            'Farmasi': {'Analisis Logis': 0.3, 'Kreativitas': 0.1, 'Komunikasi': 0.3, 'Kepemimpinan': 0.3},
            'Manajemen': {'Analisis Logis': 0.2, 'Kreativitas': 0.2, 'Komunikasi': 0.3, 'Kepemimpinan': 0.3},
            'Akuntansi': {'Analisis Logis': 0.4, 'Kreativitas': 0.1, 'Komunikasi': 0.2, 'Kepemimpinan': 0.3},
            'Ilmu Komunikasi': {'Analisis Logis': 0.2, 'Kreativitas': 0.3, 'Komunikasi': 0.4, 'Kepemimpinan': 0.1},
            'Psikologi': {'Analisis Logis': 0.2, 'Kreativitas': 0.2, 'Komunikasi': 0.4, 'Kepemimpinan': 0.2},
            'Sastra Inggris': {'Analisis Logis': 0.2, 'Kreativitas': 0.3, 'Komunikasi': 0.4, 'Kepemimpinan': 0.1},
            'Sastra Indonesia': {'Analisis Logis': 0.2, 'Kreativitas': 0.3, 'Komunikasi': 0.4, 'Kepemimpinan': 0.1},
            'Penerjemahan': {'Analisis Logis': 0.3, 'Kreativitas': 0.2, 'Komunikasi': 0.4, 'Kepemimpinan': 0.1}
        }
        return weights.get(major_name, {'Analisis Logis': 0.25, 'Kreativitas': 0.25, 'Komunikasi': 0.25, 'Kepemimpinan': 0.25})
    
    def get_major_interests(self, major_name):
        """Return typical interests for a major"""
        interests_map = {
            'Teknik Informatika': ['teknologi', 'sains', 'analisis'],
            'Kedokteran': ['sains', 'kesehatan', 'sosial'],
            'Teknik Elektro': ['teknologi', 'sains', 'analisis'],
            'Farmasi': ['sains', 'kesehatan', 'analisis'],
            'Manajemen': ['bisnis', 'sosial', 'komunikasi'],
            'Akuntansi': ['bisnis', 'analisis', 'ketelitian'],
            'Ilmu Komunikasi': ['komunikasi', 'sosial', 'seni'],
            'Psikologi': ['sosial', 'analisis', 'komunikasi'],
            'Sastra Inggris': ['seni', 'komunikasi', 'bahasa'],
            'Sastra Indonesia': ['seni', 'komunikasi', 'bahasa'],
            'Penerjemahan': ['bahasa', 'komunikasi', 'ketelitian']
        }
        return interests_map.get(major_name, [])
    
    def get_major_competencies(self, major_name):
        """Return typical competencies for a major"""
        competencies_map = {
            'Teknik Informatika': ['analisis', 'kreativitas'],
            'Kedokteran': ['analisis', 'komunikasi', 'ketelitian'],
            'Teknik Elektro': ['analisis', 'kreativitas'],
            'Farmasi': ['analisis', 'ketelitian'],
            'Manajemen': ['komunikasi', 'kepemimpinan'],
            'Akuntansi': ['analisis', 'ketelitian'],
            'Ilmu Komunikasi': ['komunikasi', 'kreativitas'],
            'Psikologi': ['komunikasi', 'analisis'],
            'Sastra Inggris': ['komunikasi', 'kreativitas'],
            'Sastra Indonesia': ['komunikasi', 'kreativitas'],
            'Penerjemahan': ['komunikasi', 'ketelitian']
        }
        return competencies_map.get(major_name, [])
    
    def generate_recommendations(self, user_data):
        """Generate career recommendations"""
        recommendations = []
        stream = user_data.get('stream', 'IPA')
        
        for major_data in self.majors_data.get(stream, []):
            major_name = major_data['name']
            score = self.calculate_match_score(user_data, major_data)
            
            recommendations.append({
                'major': major_name,
                'score': round(score, 1),
                'skills': major_data['skills'],
                'prospects': major_data['prospects'],
                'careers': self.career_paths.get(major_name, ['Various Career Opportunities']),
                'reasoning': self.generate_reasoning(user_data, major_data, score)
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:3]
    
    def generate_reasoning(self, user_data, major, score):
        """Generate reasoning for recommendation"""
        reasoning_parts = []
        
        if user_data.get('interests'):
            reasoning_parts.append("sesuai dengan minat Anda")
        
        if user_data.get('competencies'):
            reasoning_parts.append("sejalan dengan kompetensi Anda")
        
        if user_data.get('academic_scores'):
            reasoning_parts.append("relevan dengan kemampuan akademik")
        
        if user_data.get('essay_analysis'):
            reasoning_parts.append("didukung oleh esai reflektif Anda")
        
        return f"Rekomendasi {major['name']} karena " + ", ".join(reasoning_parts)
    
    def scrape_additional_info(self, major_name):
        """Scrape additional information for a major"""
        return self.scraper.scrape_university_info(major_name)
    
    def analyze_user_essays(self, essays):
        """Analyze user essays"""
        return self.essay_analyzer.analyze_essays(essays)

# ==================== STREAMLIT UI COMPONENTS ====================
def main():
    st.markdown('<h1 class="main-header">🎓 AI Career Guidance Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Sistem Bimbingan Karir Berbasis AI untuk Siswa Kelas 12**")
    
    # Initialize session state
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = CareerAI()
        st.session_state.current_step = 0
        st.session_state.user_data = {}
        st.session_state.scraped_data = {}
        st.session_state.scraping_done = False
        st.session_state.debug_mode = True
        st.session_state.essays = {}
    
    # Sidebar navigation
    st.sidebar.title("Navigasi")
    steps = ["Data Diri", "Data Akademik", "Minat & Kompetensi", "Esai Reflektif", "Hasil Rekomendasi"]
    current_step = st.sidebar.radio("Pilih Langkah:", steps, index=st.session_state.current_step)
    
    # Debug toggle
    st.session_state.debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=st.session_state.get('debug_mode', True))
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Update current step
    st.session_state.current_step = steps.index(current_step)
    
    # Render steps
    if current_step == "Data Diri":
        render_personal_info()
    elif current_step == "Data Akademik":
        render_academic_data()
    elif current_step == "Minat & Kompetensi":
        render_interests_competencies()
    elif current_step == "Esai Reflektif":
        render_essay_reflection()
    else:
        render_results()

def render_personal_info():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📝 Data Diri")
    
    with st.form("personal_info"):
        name = st.text_input("Nama Lengkap*", placeholder="Masukkan nama lengkap Anda")
        stream = st.selectbox("Peminatan*", ["IPA", "IPS", "Bahasa"])
        
        if st.form_submit_button("Simpan & Lanjut"):
            if name:
                st.session_state.user_data['name'] = name
                st.session_state.user_data['stream'] = stream
                st.success("✅ Data berhasil disimpan! Silakan lanjut ke langkah berikutnya.")
                st.session_state.current_step = 1
                st.rerun()
            else:
                st.error("❌ Harap isi nama lengkap")
    st.markdown('</div>', unsafe_allow_html=True)

def render_academic_data():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📊 Data Akademik")
    
    stream = st.session_state.user_data.get('stream', 'IPA')
    subjects = get_subjects_by_stream(stream)
    
    with st.form("academic_data"):
        st.write("Masukkan nilai rapor (skala 0-100):")
        
        scores = {}
        cols = st.columns(2)
        
        for i, subject in enumerate(subjects):
            with cols[i % 2]:
                scores[subject] = st.slider(f"Nilai {subject}", 0, 100, 75, key=f"score_{subject}")
        
        if st.form_submit_button("Simpan & Lanjut"):
            st.session_state.user_data['academic_scores'] = scores
            st.success("✅ Data akademik berhasil disimpan!")
            st.session_state.current_step = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def get_subjects_by_stream(stream):
    subjects = {
        'IPA': ['Matematika', 'Fisika', 'Kimia', 'Biologi'],
        'IPS': ['Matematika', 'Ekonomi', 'Sejarah', 'Geografi'],
        'Bahasa': ['Bahasa Indonesia', 'Bahasa Inggris', 'Sastra', 'Sejarah']
    }
    return subjects.get(stream, [])

def render_interests_competencies():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("🎯 Minat & Kompetensi")
    
    with st.form("interests_competencies"):
        st.subheader("Minat Aktivitas (1=Sangat Tidak Suka, 5=Sangat Suka)")
        
        interests = {
            "Analisis dan Problem Solving": st.slider("Analisis & Problem Solving", 1, 5, 3, key="interest_analysis"),
            "Kreativitas dan Seni": st.slider("Kreativitas & Seni", 1, 5, 3, key="interest_creativity"),
            "Komunikasi dan Sosial": st.slider("Komunikasi & Sosial", 1, 5, 3, key="interest_communication"),
            "Teknologi dan Programming": st.slider("Teknologi & Programming", 1, 5, 3, key="interest_tech")
        }
        
        st.subheader("Kompetensi Diri (1=Sangat Rendah, 5=Sangat Tinggi)")
        competencies = {
            "Analisis Logis": st.slider("Analisis Logis", 1, 5, 3, key="comp_analysis"),
            "Kreativitas": st.slider("Kreativitas", 1, 5, 3, key="comp_creativity"),
            "Komunikasi": st.slider("Komunikasi", 1, 5, 3, key="comp_communication"),
            "Kepemimpinan": st.slider("Kepemimpinan", 1, 5, 3, key="comp_leadership")
        }
        
        if st.form_submit_button("Simpan & Lanjut"):
            st.session_state.user_data['interests'] = interests
            st.session_state.user_data['competencies'] = competencies
            st.success("✅ Data berhasil disimpan!")
            st.session_state.current_step = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_essay_reflection():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("📝 Esai Reflektif")
    st.write("Jawab pertanyaan berikut untuk membantu sistem memahami minat, aspirasi, dan kepribadian Anda dengan lebih baik.")
    
    essay_questions = {
        'q1': "Apa cita-cita atau impian karir Anda? Jelaskan mengapa karir tersebut menarik bagi Anda.",
        'q2': "Deskripsikan kegiatan atau aktivitas yang paling Anda nikmati selama sekolah dan jelaskan apa yang membuatnya menyenangkan.",
        'q3': "Apa kekuatan terbesar yang Anda miliki dan bagaimana Anda berencana mengembangkannya di masa depan?"
    }
    
    with st.form("essay_form"):
        essays = {}
        
        for q_key, question in essay_questions.items():
            st.subheader(f"Pertanyaan {list(essay_questions.keys()).index(q_key) + 1}")
            st.write(question)
            essays[q_key] = st.text_area(
                f"Jawaban Anda:",
                height=150,
                key=f"essay_{q_key}",
                placeholder="Tulis jawaban Anda di sini... (minimal 50 kata untuk analisis optimal)"
            )
            st.markdown("---")
        
        if st.form_submit_button("📊 Analisis Esai & Lihat Rekomendasi"):
            # Check if at least one essay has sufficient content
            valid_essays = {k: v for k, v in essays.items() if v and len(v.split()) >= 10}
            
            if valid_essays:
                with st.spinner("🔄 Menganalisis esai Anda..."):
                    # Analyze essays
                    essay_analysis = st.session_state.ai_system.analyze_user_essays(valid_essays)
                    st.session_state.user_data['essay_analysis'] = essay_analysis
                    st.session_state.essays = valid_essays
                
                # Show analysis results
                st.success("✅ Analisis esai selesai! Berikut hasilnya:")
                
                if essay_analysis.get('overall'):
                    overall = essay_analysis['overall']
                    
                    st.markdown('<div class="essay-analysis">', unsafe_allow_html=True)
                    st.subheader("📊 Hasil Analisis Esai")
                    
                    # Sentiment
                    sentiment = overall.get('sentiment', {})
                    sentiment_class = f"sentiment-{sentiment.get('label', 'neutral').lower()}"
                    st.markdown(f"**Sentimen:** <span class='{sentiment_class}'>{sentiment.get('emoji', '')} {sentiment.get('label', 'N/A')}</span>", unsafe_allow_html=True)
                    
                    # Interests
                    if overall.get('interests'):
                        st.write(f"**Minat yang Terdeteksi:** {', '.join(overall['interests'])}")
                    
                    # Competencies
                    if overall.get('competencies'):
                        st.write(f"**Kompetensi yang Terdeteksi:** {', '.join(overall['competencies'])}")
                    
                    # Key phrases
                    if overall.get('key_phrases'):
                        st.write(f"**Kata Kunci Utama:** {', '.join(overall['key_phrases'][:5])}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.session_state.current_step = 4
                st.rerun()
            else:
                st.error("❌ Harap isi setidaknya satu pertanyaan esai dengan minimal 10 kata untuk analisis yang optimal.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_results():
    st.header("🎓 Hasil Rekomendasi Karir")
    
    if 'interests' not in st.session_state.user_data:
        st.warning("⚠️ Harap lengkapi data di langkah-langkah sebelumnya terlebih dahulu.")
        return
    
    # DEBUG PANEL
    if st.session_state.get('debug_mode', True):
        with st.expander("🔧 Debug Information", expanded=True):
            st.markdown('<div class="debug-panel">', unsafe_allow_html=True)
            st.write("**Scraping Status:**")
            st.write(f"- Scraping done: {st.session_state.get('scraping_done', 'Not set')}")
            st.write(f"- Scraped data keys: {list(st.session_state.get('scraped_data', {}).keys())}")
            st.write(f"- User stream: {st.session_state.get('user_data', {}).get('stream', 'No stream')}")
            st.write(f"- Essay analysis: {'Available' if st.session_state.get('user_data', {}).get('essay_analysis') else 'Not available'}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate recommendations
    recommendations = st.session_state.ai_system.generate_recommendations(st.session_state.user_data)
    
    # Web scraping untuk informasi tambahan
    if not st.session_state.get('scraping_done', False):
        st.info("🔍 **MEMULAI WEB SCRAPING** - Mencari informasi terbaru...")
        scraped_count = 0
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        for i, rec in enumerate(recommendations):
            progress_bar.progress((i + 1) / len(recommendations))
            with st.spinner(f"🕸️ Scraping info untuk {rec['major']}..."):
                try:
                    scraped_info = st.session_state.ai_system.scrape_additional_info(rec['major'])
                    st.session_state.scraped_data[rec['major']] = scraped_info
                    scraped_count += 1
                    
                    # Tampilkan hasil scraping langsung
                    status_container.success(f"✅ {rec['major']}: Berhasil - {len(scraped_info['description'])} karakter dari {scraped_info['source']}")
                    
                except Exception as e:
                    status_container.error(f"❌ {rec['major']}: Gagal - {str(e)}")
                
                time.sleep(2)  # Delay untuk menghormati server
        
        st.session_state.scraping_done = True
        st.balloons()
        st.success(f"🎉 Scraping selesai! Berhasil mengumpulkan info untuk {scraped_count} jurusan")
    
    st.subheader("📊 Rekomendasi Jurusan Terbaik untuk Anda:")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
            st.markdown(f"### 🥇 {i}. {rec['major']}")
            st.markdown(f"**Skor Kecocokan: {rec['score']}%**")
            st.markdown(f"**Prospek Karir:** {rec['prospects']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("🔍 Lihat Detail & Informasi Web"):
                st.write(f"**Alasan:** {rec['reasoning']}")
                
                st.write("**Keterampilan yang Dibutuhkan:**")
                for skill in rec['skills']:
                    st.write(f"• {skill}")
                
                st.write("**Peluang Karir:**")
                for career in rec['careers']:
                    st.write(f"• {career}")
                
                # Tampilkan informasi dari web scraping
                scraped_info = st.session_state.scraped_data.get(rec['major'])
                if scraped_info:
                    st.markdown('<div class="scraping-info">', unsafe_allow_html=True)
                    st.write("**🌐 Informasi dari Web:**")
                    st.write(scraped_info['description'])
                    st.write(f"*Sumber: {scraped_info['source']}*")
                    if scraped_info.get('url') and scraped_info['url'] != '#':
                        st.write(f"*Link: {scraped_info['url']}*")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("❌ Tidak ada data scraping untuk jurusan ini")
    
    # Tampilkan analisis esai jika ada
    if st.session_state.user_data.get('essay_analysis'):
        st.subheader("📝 Analisis Esai Reflektif Anda")
        
        overall = st.session_state.user_data['essay_analysis'].get('overall', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment = overall.get('sentiment', {})
            st.metric(
                label="Sentimen Esai", 
                value=f"{sentiment.get('emoji', '')} {sentiment.get('label', 'N/A')}",
                delta=f"Score: {sentiment.get('score', 0):.2f}" if sentiment.get('score') else None
            )
        
        with col2:
            interests = overall.get('interests', [])
            st.metric(
                label="Minat Terdeteksi", 
                value=len(interests)
            )
        
        with col3:
            competencies = overall.get('competencies', [])
            st.metric(
                label="Kompetensi Terdeteksi", 
                value=len(competencies)
            )
    
    # Career planning
    st.subheader("📅 Rencana Pengembangan Karir")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🎯 Jangka Pendek (1-2 tahun)**")
        st.write("• Fokus mata pelajaran prasyarat")
        st.write("• Ikuti ekstrakurikuler relevan")
        st.write("• Eksplorasi kursus online")
    
    with col2:
        st.write("**🚀 Jangka Menengah (2-4 tahun)**")
        st.write("• Persiapan masuk perguruan tinggi")
        st.write("• Cari pengalaman magang")
        st.write("• Bangun jaringan profesional")
    
    with col3:
        st.write("**🎓 Jangka Panjang (4+ tahun)**")
        st.write("• Selesaikan pendidikan tinggi")
        st.write("• Dapatkan sertifikasi")
        st.write("• Terus belajar dan beradaptasi")
    
    # ==================== EXPORT SECTION ====================
    st.subheader("💾 Export Hasil Konsultasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON Export
        st.write("**📄 Format Data (JSON)**")
        st.write("Untuk analisis data lanjutan")
        export_data = {
            'user_info': {
                'name': st.session_state.user_data.get('name'),
                'stream': st.session_state.user_data.get('stream')
            },
            'academic_scores': st.session_state.user_data.get('academic_scores', {}),
            'interests': st.session_state.user_data.get('interests', {}),
            'competencies': st.session_state.user_data.get('competencies', {}),
            'essays': st.session_state.get('essays', {}),
            'essay_analysis': st.session_state.user_data.get('essay_analysis', {}),
            'recommendations': recommendations,
            'scraped_data': st.session_state.scraped_data,
            'generated_at': datetime.now().isoformat()
        }
        
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name="hasil_konsultasi_karir.json",
            mime="application/json"
        )
    
    with col2:
        # PDF Export
        st.write("**📊 Laporan Lengkap (PDF)**")
        st.write("Untuk presentasi dan dokumentasi")
        
        if st.button("🔄 Generate PDF Report", key="generate_pdf"):
            try:
                with st.spinner("Membuat laporan PDF..."):
                    pdf_bytes = generate_pdf_report(
                        st.session_state.user_data, 
                        recommendations,
                        st.session_state.scraped_data,
                        st.session_state.user_data.get('essay_analysis')
                    )
                    
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name="laporan_konsultasi_karir.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                    
                    st.success("✅ Laporan PDF berhasil dibuat! Klik 'Download PDF' di atas.")
                    
            except Exception as e:
                st.error(f"❌ Error membuat PDF: {str(e)}")

if __name__ == "__main__":
    main()