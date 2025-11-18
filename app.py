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
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
    
    def analyze_interests(self, text):
        """Simple interest analysis based on keywords"""
        interests = []
        text_lower = text.lower()
        
        interest_keywords = {
            'teknologi': ['programming', 'coding', 'komputer', 'teknologi', 'software'],
            'sains': ['penelitian', 'eksperimen', 'sains', 'ilmiah'],
            'seni': ['seni', 'desain', 'kreatif', 'gambar'],
            'sosial': ['masyarakat', 'sosial', 'membantu', 'komunitas']
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                interests.append(interest)
        
        return interests
    
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

def render_academic_data():
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
    
    # Export option
    st.subheader("💾 Export Hasil")
    if st.button("📥 Download Hasil Konsultasi"):
        export_data = {
            'user_info': {
                'name': st.session_state.user_data.get('name'),
                'stream': st.session_state.user_data.get('stream')
            },
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