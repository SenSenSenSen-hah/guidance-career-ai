import streamlit as st
import pandas as pd
import numpy as np
import json

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
        
        self.career_paths = {
            'Teknik Informatika': ['Software Engineer', 'Data Scientist', 'Web Developer'],
            'Kedokteran': ['Dokter Umum', 'Dokter Spesialis', 'Peneliti Medis'],
            'Manajemen': ['Manajer Perusahaan', 'Entrepreneur', 'Konsultan Bisnis']
        }
    
    def generate_recommendations(self, user_data):
        recommendations = []
        stream = user_data.get('stream', 'IPA')
        
        for major in self.majors_data.get(stream, []):
            # Simple scoring based on interests and competencies
            score = 70  # Base score
            
            # Add bonus based on interests
            if user_data.get('interests'):
                interest_bonus = sum(user_data['interests'].values()) * 2
                score += min(interest_bonus, 20)
            
            # Add bonus based on competencies
            if user_data.get('competencies'):
                competency_bonus = sum(user_data['competencies'].values()) * 2
                score += min(competency_bonus, 10)
            
            score = min(score, 95)  # Cap at 95%
            
            recommendations.append({
                'major': major,
                'score': score,
                'careers': self.career_paths.get(major, ['Various Career Opportunities']),
                'reasoning': self.generate_reasoning(user_data, major, score)
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def generate_reasoning(self, user_data, major, score):
        reasoning_parts = []
        
        if user_data.get('interests'):
            reasoning_parts.append("sesuai dengan minat yang Anda tunjukkan")
        
        if user_data.get('competencies'):
            reasoning_parts.append("sejalan dengan kompetensi yang Anda miliki")
        
        if user_data.get('academic_scores'):
            reasoning_parts.append("relevan dengan kemampuan akademik Anda")
        
        if reasoning_parts:
            return f"Rekomendasi {major} karena " + ", ".join(reasoning_parts) + "."
        else:
            return f"Rekomendasi {major} berdasarkan analisis profil umum."

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
                st.success("✅ Data berhasil disimpan! Silakan lanjut ke langkah berikutnya.")
                st.session_state.current_step = 1
            else:
                st.error("❌ Harap isi nama lengkap")

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
            st.success("✅ Data akademik berhasil disimpan!")
            st.session_state.current_step = 2

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
        st.subheader("Minat Aktivitas (1=Sangat Tidak Suka, 5=Sangat Suka)")
        
        interests = {
            "Analisis dan Problem Solving": st.slider("Analisis & Problem Solving", 1, 5, 3),
            "Kreativitas dan Seni": st.slider("Kreativitas & Seni", 1, 5, 3),
            "Komunikasi dan Sosial": st.slider("Komunikasi & Sosial", 1, 5, 3),
            "Teknologi dan Programming": st.slider("Teknologi & Programming", 1, 5, 3)
        }
        
        st.subheader("Kompetensi Diri (1=Sangat Rendah, 5=Sangat Tinggi)")
        competencies = {
            "Analisis Logis": st.slider("Analisis Logis", 1, 5, 3),
            "Kreativitas": st.slider("Kreativitas", 1, 5, 3),
            "Komunikasi": st.slider("Komunikasi", 1, 5, 3),
            "Kepemimpinan": st.slider("Kepemimpinan", 1, 5, 3)
        }
        
        if st.form_submit_button("🎯 Lihat Rekomendasi"):
            st.session_state.user_data['interests'] = interests
            st.session_state.user_data['competencies'] = competencies
            st.balloons()
            st.success("✅ Data berhasil disimpan! Melihat rekomendasi...")
            st.session_state.current_step = 3

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
            st.markdown(f"### 🥇 {i}. {rec['major']}")
            
            # Progress bar
            progress_value = rec['score'] / 100
            st.progress(progress_value)
            
            st.write(f"**Skor Kecocokan: {rec['score']}%**")
            st.write(f"**Alasan:** {rec['reasoning']}")
            
            st.write("**Peluang Karir:**")
            for career in rec['careers']:
                st.write(f"• {career}")
            
            st.markdown("---")
    
    # Career planning
    st.subheader("📅 Rencana Pengembangan Karir")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🎯 Jangka Pendek (1-2 tahun)**")
        st.write("• Fokus pada mata pelajaran prasyarat")
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
    
    # Export option
    st.subheader("💾 Export Hasil")
    if st.button("📥 Download Hasil Konsultasi"):
        export_data = {
            'user_info': {
                'name': st.session_state.user_data.get('name'),
                'stream': st.session_state.user_data.get('stream')
            },
            'academic_scores': st.session_state.user_data.get('academic_scores', {}),
            'interests': st.session_state.user_data.get('interests', {}),
            'competencies': st.session_state.user_data.get('competencies', {}),
            'recommendations': recommendations,
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        st.download_button(
            label="📄 Download JSON File",
            data=json.dumps(export_data, indent=2),
            file_name="hasil_konsultasi_karir.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()