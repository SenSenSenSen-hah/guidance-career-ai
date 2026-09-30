import streamlit as st
import google.generativeai as genai
import json
import os
import sqlite3
import base64
from datetime import datetime
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS

# ==================== 1. KONFIGURASI SISTEM ====================
st.set_page_config(page_title="Autonomous Career Agent (RAG)", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; font-weight: 800; }
    .xai-box { border-left: 4px solid #ff9800; background-color: #fff3e0; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

DB_JSON_FILE = 'agent_knowledge_base.json'
DB_SQLITE_FILE = 'skripsi_logs.db'

# ==================== 2. INITIALIZATION (DATABASE & RAG) ====================

def init_databases():
    # A. JSON Knowledge Base (Untuk RAG)
    if not os.path.exists(DB_JSON_FILE):
        default_db = [
            {"jurusan": "Teknik Informatika", "riasec": "IRC", "deskripsi": "Fokus pada komputasi, algoritma, kecerdasan buatan, dan pemrograman perangkat lunak. Cocok untuk yang suka memecahkan teka-teki logika."},
            {"jurusan": "Psikologi", "riasec": "SIA", "deskripsi": "Mempelajari perilaku manusia dan proses mental. Memerlukan empati tinggi dan kemampuan observasi analitis."},
            {"jurusan": "Manajemen Bisnis", "riasec": "EAS", "deskripsi": "Pengelolaan organisasi, kepemimpinan, dan finansial. Cocok untuk yang berjiwa wirausaha dan suka bernegosiasi."},
            {"jurusan": "Ilmu Komunikasi", "riasec": "SAE", "deskripsi": "Fokus pada jurnalistik, public relations, dan media kreatif. Membutuhkan kemampuan verbal dan sosialisasi yang baik."},
            {"jurusan": "Kedokteran", "riasec": "ISA", "deskripsi": "Ilmu medis, kesehatan, dan penyembuhan pasien. Sangat butuh ketelitian sains dan keinginan membantu orang lain."},
            {"jurusan": "Desain Komunikasi Visual", "riasec": "AES", "deskripsi": "Seni terapan, desain grafis, dan kreativitas visual. Menggabungkan teknologi dan rasa estetika tinggi."}
        ]
        with open(DB_JSON_FILE, 'w') as f:
            json.dump(default_db, f, indent=4)
            
    # B. SQLite Database (Untuk Logging/Skripsi)
    conn = sqlite3.connect(DB_SQLITE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, tanggal TEXT, nama_siswa TEXT, riasec_code TEXT, jurusan_rekomendasi TEXT, ringkasan_chat TEXT)''')
    conn.commit()
    conn.close()

init_databases()

# ==================== 3. TOOLS UNTUK AUTONOMOUS AGENT ====================

def search_major_with_rag(riasec_code: str, student_story: str) -> str:
    """
    ALAT 1 (RAG DATABASE SEARCH): Gunakan alat ini untuk mencari rekomendasi jurusan di database. 
    Masukkan kode RIASEC (misal: 'IRE') dan ringkasan cerita/minat siswa (misal: 'suka main komputer').
    """
    try:
        with open(DB_JSON_FILE, 'r') as f:
            db = json.load(f)
            
        # Filter 1: Cek irisan huruf RIASEC
        filtered_db = []
        user_set = set(riasec_code.upper())
        for item in db:
            if len(user_set.intersection(set(item['riasec']))) >= 1:
                filtered_db.append(item)
                
        if not filtered_db:
            filtered_db = db # Fallback jika tidak ada yang cocok sama sekali
            
        # Filter 2: RAG (Semantic Search menggunakan TF-IDF & Cosine Similarity)
        # Mengukur kedekatan cerita siswa dengan deskripsi kurikulum jurusan
        corpus = [item['deskripsi'] for item in filtered_db]
        corpus.append(student_story) # Masukkan cerita siswa di akhir untuk dibandingkan
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Hitung kemiripan cerita siswa (index terakhir) dengan semua jurusan
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
        # Urutkan berdasarkan skor tertinggi
        ranked_indices = cosine_sim.argsort()[::-1]
        
        hasil = ["HASIL PENCARIAN DATABASE (Diurutkan dari paling relevan):"]
        for idx in ranked_indices[:3]: # Ambil Top 3
            item = filtered_db[idx]
            skor = round(cosine_sim[idx] * 100, 1)
            hasil.append(f"- {item['jurusan']} (Kode: {item['riasec']}, Skor Relevansi Semantik: {skor}%): {item['deskripsi']}")
            
        return "\n".join(hasil)
    except Exception as e:
        return f"Gagal mencari database: {str(e)}"

def search_internet_job_prospects(query: str) -> str:
    """
    ALAT 2 (WEB SEARCH): Gunakan alat ini JIKA pengguna bertanya tentang prospek kerja, gaji, atau info tren karir masa depan yang butuh data internet real-time.
    """
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "Tidak ditemukan informasi di internet."
        
        rangkuman = []
        for r in results:
            rangkuman.append(f"Sumber: {r.get('title')}\nInfo: {r.get('body')}\n")
        return "\n".join(rangkuman)
    except Exception as e:
        return "Gagal mengakses internet saat ini."

def generate_pdf_and_log_data(nama_siswa: str, riasec_code: str, nama_jurusan: str, alasan_rekomendasi: str) -> str:
    """
    ALAT 3 (PDF & LOGGING): PANGGIL ALAT INI DI AKHIR PERCAKAPAN saat kamu sudah yakin memberikan rekomendasi final.
    Fungsi ini akan menyimpan data ke database SQLite untuk penelitian, dan membuat file PDF untuk didownload pengguna.
    """
    try:
        # 1. Simpan ke SQLite Logging (Sangat berguna untuk Bab 4 Skripsi)
        conn = sqlite3.connect(DB_SQLITE_FILE)
        c = conn.cursor()
        tgl = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_logs (tanggal, nama_siswa, riasec_code, jurusan_rekomendasi, ringkasan_chat) VALUES (?, ?, ?, ?, ?)",
                  (tgl, nama_siswa, riasec_code, nama_jurusan, alasan_rekomendasi))
        conn.commit()
        conn.close()

        # 2. Buat File PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "LAPORAN KONSULTASI KARIR AI", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Nama: {nama_siswa.encode('latin-1', 'replace').decode('latin-1')}", ln=1)
        pdf.cell(0, 10, f"Kode Kepribadian Holland (RIASEC): {riasec_code}", ln=1)
        pdf.cell(0, 10, f"Rekomendasi Utama: {nama_jurusan.encode('latin-1', 'replace').decode('latin-1')}", ln=1)
        pdf.ln(5)
        
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 7, alasan_rekomendasi.encode('latin-1', 'replace').decode('latin-1'))
        
        # Simpan ke Session State untuk ditampilkan sebagai tombol download di UI
        st.session_state['pdf_data'] = pdf.output(dest='S').encode('latin-1', 'replace')
        st.session_state['pdf_filename'] = f"Hasil_Karir_{nama_siswa.replace(' ', '_')}.pdf"
        
        return "SUKSES: Data berhasil dicatat ke database penelitian dan PDF telah disiapkan untuk diunduh oleh pengguna."
    except Exception as e:
        return f"GAGAL membuat PDF atau menyimpan ke database: {str(e)}"

# ==================== 4. INISIALISASI AUTONOMOUS AGENT ====================

def get_agent():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        system_instruction = """
        Kamu adalah Autonomous Career Agent. Kamu memiliki 3 ALAT (Tools):
        1. search_major_with_rag: Untuk mencari jurusan.
        2. search_internet_job_prospects: Untuk mencari info gaji/prospek kerja di internet.
        3. generate_pdf_and_log_data: Untuk membuat laporan PDF dan menyimpan data ke database.

        Tugasmu:
        1. Sapa pengguna, tanyakan namanya, lalu gali hobi dan minatnya.
        2. Analisis Kode RIASEC (3 huruf dominan).
        3. Panggil alat `search_major_with_rag` untuk mencari jurusan.
        4. Jika pengguna bertanya soal prospek kerja, panggil `search_internet_job_prospects`.
        5. SAAT PERCAKAPAN HAMPIR SELESAI, tawarkan untuk membuatkan Laporan PDF. Jika ia mau, panggil alat `generate_pdf_and_log_data`.
        6. Berbicara dengan bahasa Indonesia yang santai dan empatik.
        """
        
        # Menggunakan model Pro agar logika pemanggilan Tools (Function Calling) sangat cerdas
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=[search_major_with_rag, search_internet_job_prospects, generate_pdf_and_log_data],
            system_instruction=system_instruction
        )
        return model
    except Exception as e:
        st.error(f"Error Konfigurasi API: {e}")
        return None

# ==================== 5. ANTARMUKA CHAT & XAI (EXPLAINABLE AI) ====================

st.markdown('<h1 class="main-header">🤖 Autonomous Career Agent (RAG + Web Tools)</h1>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "parts": ["Halo! Saya AI Konselor Karir otonom. Boleh tahu siapa namamu dan apa aktivitas yang paling kamu nikmati akhir-akhir ini?"]}]

if "chat_session" not in st.session_state:
    model = get_agent()
    if model:
        st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)

# Tampilkan riwayat obrolan (Teks saja)
for msg in st.session_state.messages:
    if "parts" in msg:
        with st.chat_message("ai" if msg["role"] == "model" else "human"):
            st.write(msg["parts"][0])

user_input = st.chat_input("Ketik di sini...")

if user_input:
    with st.chat_message("human"):
        st.write(user_input)
    st.session_state.messages.append({"role": "human", "parts": [user_input]})
    
    with st.chat_message("ai"):
        with st.spinner("🤖 Agen sedang berpikir, menganalisis, dan memanggil alat jika diperlukan..."):
            try:
                # Rekam panjang history sebelum dikirim (Untuk melacak Chain of Thought)
                old_history_len = len(st.session_state.chat_session.history)
                
                # Kirim pesan (Proses otonom berjalan di sini)
                response = st.session_state.chat_session.send_message(user_input)
                
                # ==========================================
                # FITUR XAI (Explainable AI / Transparansi Pikiran)
                # ==========================================
                new_history = st.session_state.chat_session.history[old_history_len:]
                tools_used = []
                for h_msg in new_history:
                    for part in h_msg.parts:
                        # Mengecek apakah agen memanggil fungsi Python di balik layar
                        if hasattr(part, 'function_call') and part.function_call:
                            fn_name = part.function_call.name
                            args = dict(part.function_call.args)
                            tools_used.append(f"🔧 <b>Agent Action:</b> Memanggil <code>{fn_name}</code> dengan argumen {args}")
                
                # Menampilkan proses berpikir AI jika ada alat yang dipanggil
                if tools_used:
                    with st.expander("🧠 Lihat Proses Berpikir AI (Chain-of-Thought)"):
                        for t in tools_used:
                            st.markdown(f'<div class="xai-box">{t}</div>', unsafe_allow_html=True)
                
                # ==========================================
                
                # Tampilkan balasan teks
                st.write(response.text)
                st.session_state.messages.append({"role": "model", "parts": [response.text]})
                
            except Exception as e:
                st.error(f"Terjadi kesalahan agen: {str(e)}")

# Menampilkan tombol download PDF jika agen sudah mengeksekusi alat generate_pdf
if 'pdf_data' in st.session_state:
    st.markdown("---")
    st.success("📄 Agen telah membuat laporan PDF Anda!")
    st.download_button(
        label="📥 Download Laporan Karir (PDF)",
        data=st.session_state['pdf_data'],
        file_name=st.session_state['pdf_filename'],
        mime="application/pdf"
    )
    
# Fitur Admin Tersembunyi (Untuk Skripsi Bab 4)
with st.sidebar:
    st.markdown("### ⚙️ Admin & Riset")
    if st.checkbox("Buka Panel Peneliti"):
        st.write("Panel ini digunakan untuk melihat data eksperimen.")
        if os.path.exists(DB_SQLITE_FILE):
            conn = sqlite3.connect(DB_SQLITE_FILE)
            import pandas as pd
            df = pd.read_sql_query("SELECT * FROM chat_logs", conn)
            st.dataframe(df)
            conn.close()
