import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from datetime import datetime
from fpdf import FPDF

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
</style>
""", unsafe_allow_html=True)

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
    
    def add_recommendation_card(self, rank, major, score, reasoning, careers):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, f'{rank}. {major} - Skor: {score}%', 0, 1)
        
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, f'Alasan: {reasoning}')
        
        self.set_font('Arial', 'I', 10)
        self.cell(0, 6, 'Peluang Karir:', 0, 1)
        for career in careers:
            self.cell(10)  # indent
            self.cell(0, 6, f'• {career}', 0, 1)
        
        self.ln(5)

def generate_pdf_report(user_data, recommendations):
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
    
    # Recommendations
    pdf.add_section_title('🎓 Rekomendasi Jurusan')
    for i, rec in enumerate(recommendations, 1):
        pdf.add_recommendation_card(
            i, rec['major'], rec['score'], 
            rec['reasoning'], rec['careers']
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
            'Teknik Informatika': ['Software Engineer', 'Data Scientist', 'Web Developer'],
            'Kedokteran': ['Dokter Umum', 'Dokter Spesialis', 'Peneliti Medis'],
            'Manajemen': ['Manajer Perusahaan', 'Entrepreneur', 'Konsultan Bisnis'],
            'Psikologi': ['Psikolog Klinis', 'HR Specialist', 'Konselor']
        }
    
    def calculate_match_score(self, user_data, major):
        """Calculate match score between user and major"""
        score = 60  # Base score
        
        # Academic match (simplified)
        if user_data.get('academic_scores'):
            score += 10
        
        # Interest match
        if user_data.get('interests'):
            interest_score = sum(user_data['interests'].values()) * 1.5
            score += min(interest_score, 20)
        
        # Competency match
        if user_data.get('competencies'):
            competency_score = sum(user_data['competencies'].values()) * 1.5
            score += min(competency_score, 10)
        
        return min(score, 95)
    
    def generate_recommendations(self, user_data):
        """Generate career recommendations"""
        recommendations = []
        stream = user_data.get('stream', 'IPA')
        
        for major_data in self.majors_data.get(stream, []):
            major_name = major_data['name']
            score = self.calculate_match_score(user_data, major_data)
            
            recommendations.append({
                'major': major_name,
                'score': score,
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
        
        return f"Rekomendasi {major['name']} karena " + ", ".join(reasoning_parts)

# ==================== STREAMLIT UI COMPONENTS ====================
def main():
    st.markdown('<h1 class="main-header">🎓 AI Career Guidance Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Sistem Bimbingan Karir Berbasis AI untuk Siswa Kelas 12**")
    
    # Initialize AI system
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = CareerAI()
        st.session_state.current_step = 0
        st.session_state.user_data = {}
    
    # Sidebar navigation
    st.sidebar.title("Navigasi")
    steps = ["Data Diri", "Data Akademik", "Minat & Kompetensi", "Hasil Rekomendasi"]
    current_step = st.sidebar.radio("Pilih Langkah:", steps, index=st.session_state.current_step)
    
    # Update current step
    st.session_state.current_step = steps.index(current_step)
    
    # Render steps
    if current_step == "Data Diri":
        render_personal_info()
    elif current_step == "Data Akademik":
        render_academic_data()
    elif current_step == "Minat & Kompetensi":
        render_interests_competencies()
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
        
        if st.form_submit_button("🎯 Lihat Rekomendasi"):
            st.session_state.user_data['interests'] = interests
            st.session_state.user_data['competencies'] = competencies
            st.balloons()
            st.success("✅ Data berhasil disimpan! Melihat rekomendasi...")
            st.session_state.current_step = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_results():
    st.header("🎓 Hasil Rekomendasi Karir")
    
    if 'interests' not in st.session_state.user_data:
        st.warning("⚠️ Harap lengkapi data di langkah-langkah sebelumnya terlebih dahulu.")
        return
    
    # Generate recommendations
    recommendations = st.session_state.ai_system.generate_recommendations(st.session_state.user_data)
    
    st.subheader("📊 Rekomendasi Jurusan Terbaik untuk Anda:")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
            st.markdown(f"### 🥇 {i}. {rec['major']}")
            st.markdown(f"**Skor Kecocokan: {rec['score']}%**")
            st.markdown(f"**Prospek Karir:** {rec['prospects']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("🔍 Lihat Detail"):
                st.write(f"**Alasan:** {rec['reasoning']}")
                
                st.write("**Keterampilan yang Dibutuhkan:**")
                for skill in rec['skills']:
                    st.write(f"• {skill}")
                
                st.write("**Peluang Karir:**")
                for career in rec['careers']:
                    st.write(f"• {career}")
    
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
            'recommendations': recommendations,
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
                        recommendations
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