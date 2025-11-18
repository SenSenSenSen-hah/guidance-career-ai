import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleCareerAI:
    def __init__(self):
        self.majors_data = {
            'IPA': ['Teknik Informatika', 'Kedokteran', 'Teknik Elektro', 'Farmasi', 'Arsitektur'],
            'IPS': ['Manajemen', 'Akuntansi', 'Ilmu Komunikasi', 'Psikologi', 'Hukum'],
            'Bahasa': ['Sastra Inggris', 'Sastra Indonesia', 'Penerjemahan', 'Ilmu Perpustakaan']
        }
    
    def generate_recommendations(self, user_data):
        recommendations = []
        stream = user_data.get('stream', 'IPA')
        
        for major in self.majors_data.get(stream, []):
            score = np.random.randint(70, 96)  # Simulasi scoring
            recommendations.append({
                'major': major,
                'score': score,
                'reasoning': f"Rekomendasi berdasarkan minat dan kemampuan di peminatan {stream}"
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)

def main():
    st.markdown('<h1 class="main-header">🎓 AI Career Guidance Platform</h1>', unsafe_allow_html=True)
    st.markdown("**Sistem Bimbingan Karir Berbasis AI untuk Siswa Kelas 12**")
    
    # Initialize AI system
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = SimpleCareerAI()
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
    st.header("📝 Data Diri")
    
    with st.form("personal_info"):
        name = st.text_input("Nama Lengkap*", placeholder="Masukkan nama lengkap Anda")
        stream = st.selectbox("Peminatan*", ["IPA", "IPS", "Bahasa"])
        
        if st.form_submit_button("Simpan & Lanjut"):
            if name:
                st.session_state.user_data['name'] = name
                st.session_state.user_data['stream'] = stream
                st.success("Data berhasil disimpan! Silakan lanjut ke langkah berikutnya.")
            else:
                st.error("Harap isi nama lengkap")

def render_academic_data():
    st.header("📊 Data Akademik")
    
    stream = st.session_state.user_data.get('stream', 'IPA')
    subjects = get_subjects_by_stream(stream)
    
    with st.form("academic_data"):
        st.write("Masukkan nilai rapor (skala 0-100):")
        
        scores = {}
        for subject in subjects:
            scores[subject] = st.slider(f"Nilai {subject}", 0, 100, 75)
        
        if st.form_submit_button("Simpan & Lanjut"):
            st.session_state.user_data['academic_scores'] = scores
            st.success("Data akademik berhasil disimpan!")

def get_subjects_by_stream(stream):
    subjects = {
        'IPA': ['Matematika', 'Fisika', 'Kimia', 'Biologi'],
        'IPS': ['Matematika', 'Ekonomi', 'Sejarah', 'Geografi'],
        'Bahasa': ['Bahasa Indonesia', 'Bahasa Inggris', 'Sastra', 'Sejarah']
    }
    return subjects.get(stream, [])

def render_interests_competencies():
    st.header("🎯 Minat & Kompetensi")
    
    with st.form("interests_competencies"):
        st.subheader("Minat Aktivitas")
        interests = {
            "Analisis dan Problem Solving": st.slider("Analisis & Problem Solving", 1, 5, 3),
            "Kreativitas dan Seni": st.slider("Kreativitas & Seni", 1, 5, 3),
            "Komunikasi dan Sosial": st.slider("Komunikasi & Sosial", 1, 5, 3),
            "Teknologi dan Programming": st.slider("Teknologi & Programming", 1, 5, 3)
        }
        
        st.subheader("Kompetensi Diri")
        competencies = {
            "Analisis Logis": st.slider("Analisis Logis", 1, 5, 3),
            "Kreativitas": st.slider("Kreativitas", 1, 5, 3),
            "Komunikasi": st.slider("Komunikasi", 1, 5, 3),
            "Kepemimpinan": st.slider("Kepemimpinan", 1, 5, 3)
        }
        
        if st.form_submit_button("Lihat Rekomendasi"):
            st.session_state.user_data['interests'] = interests
            st.session_state.user_data['competencies'] = competencies
            st.balloons()
            st.success("Data berhasil disimpan! Silakan lihat rekomendasi di langkah terakhir.")

def render_results():
    st.header("🎓 Hasil Rekomendasi Karir")
    
    if 'interests' not in st.session_state.user_data:
        st.warning("Harap lengkapi data di langkah-langkah sebelumnya terlebih dahulu.")
        return
    
    # Generate recommendations
    recommendations = st.session_state.ai_system.generate_recommendations(st.session_state.user_data)
    
    st.subheader("Rekomendasi Jurusan Terbaik untuk Anda:")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f"### {i}. {rec['major']}")
            st.progress(rec['score']/100)
            st.write(f"**Skor Kecocokan: {rec['score']}%**")
            st.write(f"*{rec['reasoning']}*")
            st.markdown("---")
    
    # Career planning
    st.subheader("📅 Rencana Pengembangan Karir")
    st.write("""
    1. **Jangka Pendek (1-2 tahun):**
       - Fokus pada mata pelajaran prasyarat
       - Ikuti ekstrakurikuler yang relevan
       - Eksplorasi melalui kursus online
    
    2. **Jangka Menengah (2-4 tahun):**
       - Persiapan masuk perguruan tinggi
       - Cari pengalaman magang atau volunteering
       - Bangun jaringan profesional
    
    3. **Jangka Panjang (4+ tahun):**
       - Selesaikan pendidikan tinggi
       - Dapatkan sertifikasi profesional
       - Terus belajar dan beradaptasi
    """)
    
    # Export option
    st.subheader("💾 Export Hasil")
    if st.button("Download Hasil Konsultasi"):
        export_data = {
            'user_info': st.session_state.user_data,
            'recommendations': recommendations,
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        st.download_button(
            label="Download JSON File",
            data=json.dumps(export_data, indent=2),
            file_name="hasil_konsultasi_karir.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()