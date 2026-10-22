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
from tavily import TavilyClient

#1. KONFIGURASI SISTEM
st.set_page_config(
    page_title="Autonomous Career Agent (RAG)", 
    initial_sidebar_state="expanded", 
    layout="wide"
)
st.sidebar.success("SIDEBAR")

st.markdown("""
    <style>
    [data-testid="stToolbar"] {visibility: hidden !important;}
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; font-weight: 800; }
    .xai-box { border-left: 4px solid #ff9800; background-color: #fff3e0; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.85em; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

DB_JSON_FILE = 'agent_knowledge_base.json'
DB_SQLITE_FILE = 'skripsi_logs.db'

# 2. DATABASE & RAG

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
            
    # B. SQLite Database
    conn = sqlite3.connect(DB_SQLITE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, tanggal TEXT, nama_siswa TEXT, riasec_code TEXT, jurusan_rekomendasi TEXT, ringkasan_chat TEXT)''')
    conn.commit()
    conn.close()

init_databases()

# 3. TOOLS UNTUK AUTONOMOUS AGENT

def search_major_with_rag(riasec_code: str, student_story: str) -> str:
    """
    ALAT 1 (RAG DATABASE SEARCH): Gunakan alat ini untuk mencari rekomendasi jurusan di database. 
    Masukkan kode RIASEC (misal: 'IRE') dan ringkasan cerita/minat siswa (misal: 'suka main komputer').
    """
    try:
        import streamlit as st
        
        with open(DB_JSON_FILE, 'r') as f:
            db = json.load(f)
            
        # Filter 1: Cek irisan huruf RIASEC
        filtered_db = []
        user_set = set(riasec_code.upper())
        for item in db:
            # Ambil nilai kode RIASEC (Kompatibel untuk 'kode_riasec' maupun 'riasec')
            kode_item = item.get('kode_riasec') or item.get('riasec') or ''
            
            if len(user_set.intersection(set(kode_item.upper()))) >= 1:
                filtered_db.append(item)
                
        if not filtered_db:
            filtered_db = db # Fallback jika tidak ada yang cocok sama sekali
            
        # Filter 2: RAG (Semantic Search menggunakan TF-IDF & Cosine Similarity)
        corpus = [item.get('deskripsi', '') for item in filtered_db]
        corpus.append(student_story) 
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        ranked_indices = cosine_sim.argsort()[::-1]
        
        hasil = ["HASIL PENCARIAN DATABASE (Diurutkan dari paling relevan):"]
        for idx in ranked_indices[:3]: # Ambil Top 3
            item = filtered_db[idx]
            skor = round(cosine_sim[idx] * 100, 1)
            
            # Ambil nama jurusan dan kode riasec
            nama_jurusan = item.get('nama_jurusan') or item.get('jurusan') or 'Nama Tidak Diketahui'
            kode_jurusan = item.get('kode_riasec') or item.get('riasec') or '---'
            deskripsi = item.get('deskripsi', '')
            
            hasil.append(f"- {nama_jurusan} (Kode: {kode_jurusan}, Skor Relevansi Semantik: {skor}%): {deskripsi}")
            
        return "\n".join(hasil)
        
    except Exception as e:
        st.error(f"DEBUG ERROR PADA ALAT RAG: {type(e).__name__} - {str(e)}")
        return f"Gagal mencari database karena masalah teknis: {str(e)}"
        
def search_internet_job_prospects(query: str) -> str:
    """
    ALAT 2 (WEB SEARCH): Gunakan alat ini JIKA pengguna bertanya tentang prospek kerja, gaji, atau info tren karir masa depan yang butuh data internet real-time.
    """
    try:
        # API KEY
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            import streamlit as st
            api_key = st.secrets["TAVILY_API_KEY"]
            
        # Tavily API
        tavily = TavilyClient(api_key=api_key)
        
        # Pencarian ke internet
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        
        if not response.get('results'):
            return "Tidak ditemukan informasi di internet."
            
        # Merangkum hasil
        rangkuman = []
        for r in response.get('results', []):
            rangkuman.append(f"Sumber: {r.get('title')}\nInfo: {r.get('content')}\n")
        return "\n".join(rangkuman)
        
    except Exception as e:
        print(f"Error Tavily: {e}")
        return "Gagal mengakses internet saat ini karena kendala jaringan."
        
def generate_pdf_and_log_data(
    nama_siswa: str, 
    riasec_code: str, 
    nama_jurusan: str, 
    alasan_rekomendasi: str,
    riasec_detail: str,          
    prospek_kerja: str,          
    estimasi_gaji: str,          
    hal_yang_dikembangkan: str   
) -> str:
    """
    ALAT 3 (PDF & LOGGING): PANGGIL ALAT INI DI AKHIR PERCAKAPAN saat kamu sudah yakin memberikan rekomendasi final.
    Fungsi ini akan menyimpan data ke database SQLite untuk penelitian, dan membuat file PDF untuk didownload pengguna.
    PASTIKAN kamu mengisi parameter riasec_detail, prospek_kerja, estimasi_gaji, dan hal_yang_dikembangkan secara lengkap dari hasil analisismu dan pencarian web.
    """
    try:
        # 1. Simpan ke SQLite
        conn = sqlite3.connect(DB_SQLITE_FILE)
        c = conn.cursor()
        tgl = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_logs (tanggal, nama_siswa, riasec_code, jurusan_rekomendasi, ringkasan_chat) VALUES (?, ?, ?, ?, ?)",
                  (tgl, nama_siswa, riasec_code, nama_jurusan, alasan_rekomendasi))
        conn.commit()
        conn.close()

        # Fungsi bantuan agar teks tidak error saat diubah ke PDF
        def clean_text(text: str) -> str:
            if not text:
                return "-"
            
            text = text.replace('–', '-')  # Ganti en-dash
            text = text.replace('—', '-')  # Ganti em-dash
            text = text.replace('“', '"').replace('”', '"')  # Ganti tanda kutip pintar
            text = text.replace('‘', "'").replace('’', "'")  # Ganti tanda kutip tunggal pintar
            text = text.replace('•', '-')  # Ganti simbol bullet point
            
            return text.encode('latin-1', 'replace').decode('latin-1')
            
        # 2. Buat File PDF
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "LAPORAN KONSULTASI KARIR AI", 0, 1, 'C')
        pdf.ln(5)
        
        # --- DATA DIRI & HASIL RIASEC ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Nama: {clean_text(nama_siswa)}", ln=1)
        pdf.cell(0, 10, f"Kode Kepribadian Holland (RIASEC): {riasec_code}", ln=1)
        
        pdf.set_font("Arial", '', 11)
        # detail RIASEC
        pdf.multi_cell(0, 7, clean_text(f"Detail Kepribadian:\n{riasec_detail}"))
        pdf.ln(3)
        
        # Rekomendasi Jurusan
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Rekomendasi Utama: {clean_text(nama_jurusan)}", ln=1)
        
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 7, clean_text(alasan_rekomendasi))
        pdf.ln(5)
        
        # Prospek kerja & gaji
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Prospek Kerja & Estimasi Gaji:", ln=1)
        
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 7, clean_text(f"Prospek Kerja:\n{prospek_kerja}"))
        pdf.ln(2)
        pdf.multi_cell(0, 7, clean_text(f"Estimasi Gaji Saat Ini:\n{estimasi_gaji}"))
        pdf.ln(5)

        # Hal yang perlu dikembangkan
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Hal yang Perlu Dikembangkan:", ln=1)
        
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 7, clean_text(hal_yang_dikembangkan))
        
        # --- OUTPUT ---
        st.session_state['pdf_data'] = pdf.output(dest='S').encode('latin-1', 'replace')
        st.session_state['pdf_filename'] = f"Laporan_Karir_{nama_siswa.replace(' ', '_')}.pdf"
        
        return "SUKSES: Data berhasil dicatat ke database penelitian dan PDF telah disiapkan untuk diunduh oleh pengguna."
    except Exception as e:
        return f"GAGAL membuat PDF atau menyimpan ke database: {str(e)}"
        
# 4. INISIALISASI AUTONOMOUS AGENT

def get_agent():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # 1. Ask Google AI Studio directly which models this key can use
        all_models = list(genai.list_models())
        valid_models = [
            m.name for m in all_models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not valid_models:
            st.error("API Key connects, but no models supporting 'generateContent' were returned.")
            return None

        # 2. Prefer flash/pro models from the active list returned by Google
        chosen_model = None
        preference_order = [
            'models/gemini-2.0-flash',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-1.5-flash-latest'
        ]
        
        for pref in preference_order:
            if pref in valid_models:
                chosen_model = pref
                break
                
        if not chosen_model:
            chosen_model = valid_models[0]

        system_instruction = """
        
        Kamu adalah Autonomous Career Agent yang cerdas dan empatik. 
        Kamu memiliki 3 ALAT (Tools) utama:
    1. search_major_with_rag: Untuk mencari dan mencocokkan jurusan berdasarkan minat.
    2. search_internet_job_prospects: Untuk mencari data faktual terkini mengenai gaji, tren, atau prospek kerja di internet.
    3. generate_pdf_and_log_data: Untuk membuat laporan PDF dan menyimpan data akhir ke database.

    ATURAN MUTLAK PENGGUNAAN ALAT (TOOLS):
    - Kamu DILARANG KERAS menebak, mengarang, atau menggunakan pengetahuan bawaanmu untuk menjawab pertanyaan seputar gaji, tren masa depan, dan prospek karir. 
    - JIKA pengguna menyinggung, bertanya, atau penasaran tentang GAJI, PROSPEK KERJA, atau TREN INDUSTRI, kamu WAJIB LANGSUNG MEMANGGIL alat `search_internet_job_prospects`. Jangan pernah menjawab tanpa data dari alat ini!

    Langkah Tugasmu:
    1. Sapa pengguna, tanyakan namanya, lalu gali hobi dan minatnya secara interaktif.
    2. Analisis Kode RIASEC (3 huruf dominan) dari percakapan tersebut.
    3. WAJIB panggil alat `search_major_with_rag` untuk mencari rekomendasi jurusan yang tepat.
    4. JIKA pengguna bertanya soal prospek kerja atau gaji, panggil alat `search_internet_job_prospects` untuk mencari fakta web (Patuhi Aturan Mutlak di atas).
    5. SAAT PERCAKAPAN HAMPIR SELESAI, tawarkan untuk membuatkan Laporan PDF. JIKA pengguna setuju, rangkum seluruh data (termasuk kepanjangan detail 3 huruf dominan RIASEC, daftar prospek kerja, estimasi gaji, dan hal yang perlu dikembangkan) lalu WAJIB panggil alat `generate_pdf_and_log_data` dengan argumen yang lengkap.
    6. Berkomunikasi selalu menggunakan bahasa Indonesia yang santai, empatik, suportif, dan memotivasi siswa.

    Aturan tambahan yang wajib dipatuhi:
    1. Jika pengguna tidak menyebutkan namanya, sapa dengan 'Siswa' atau 'Teman'. DILARANG KERAS mengarang nama, menebak nama, atau menyebut diri Anda dengan nama tertentu.
    2. Jika dokumen (RAG) menampilkan beberapa jurusan, sebutkan semua jurusan tersebut kepada pengguna dan jelaskan perbedaan fokusnya agar pengguna bisa memilih.
    3. Jika alat pencarian RAG mengembalikan lebih dari satu jurusan (misal: Sosiologi dan Antropologi), Anda wajib menyebutkan semuanya dan menjelaskan perbedaan fokusnya kepada siswa.
        """

        model = genai.GenerativeModel(
            model_name=chosen_model,
            tools=[search_major_with_rag, search_internet_job_prospects, generate_pdf_and_log_data],
            system_instruction=system_instruction
        )
        return model

    except Exception as e:
        st.error(f"Error Konfigurasi API: {e}")
        return None
        
# Interface & XAI

st.markdown('<h1 class="main-header">Autonomous Career Agent (RAG + Web Tools)</h1>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "parts": ["Halo! Saya AI Konselor Karir. Boleh tahu siapa namamu dan apa aktivitas yang paling kamu nikmati akhir-akhir ini?"]}]

if "chat_session" not in st.session_state:
    model = get_agent()
    if model:
        st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)

# Tampilkan riwayat obrolan
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
                # Rekam panjang history sebelum dikirim
                old_history_len = len(st.session_state.chat_session.history)
                
                # Kirim pesan
                response = st.session_state.chat_session.send_message(user_input)
                
                # FITUR XAI (Explainable AI / Transparansi Pikiran)
                
                new_history = st.session_state.chat_session.history[old_history_len:]
                tools_used = []
                for h_msg in new_history:
                    for part in h_msg.parts:
                        
                        if hasattr(part, 'function_call') and part.function_call:
                            fn_name = part.function_call.name
                            args = dict(part.function_call.args)
                            tools_used.append(f"🔧 <b>Agent Action:</b> Memanggil <code>{fn_name}</code> dengan argumen {args}")
                
                # Menampilkan proses berpikir AI 
                if tools_used:
                    with st.expander("🧠 Lihat Proses Berpikir AI (Chain-of-Thought)"):
                        for t in tools_used:
                            st.markdown(f'<div class="xai-box">{t}</div>', unsafe_allow_html=True)
                
                # Tampilkan balasan teks
                st.write(response.text)
                st.session_state.messages.append({"role": "model", "parts": [response.text]})
                
            except Exception as e:
                st.error(f"Terjadi kesalahan agen: {str(e)}")

#tombol download PDF
if 'pdf_data' in st.session_state:
    st.markdown("---")
    st.success("📄 Agen telah membuat laporan PDF Anda!")
    st.download_button(
        label="📥 Download Laporan Karir (PDF)",
        data=st.session_state['pdf_data'],
        file_name=st.session_state['pdf_filename'],
        mime="application/pdf"
    )
    
with st.sidebar:
    st.markdown("### ⚙️ Admin & Riset")
    
    # password
    kunci_admin = st.text_input("Masukkan Sandi Peneliti:", type="password")
    
    if kunci_admin == st.secrets["ADMIN_PASSWORD"]:
        
        if st.checkbox("Buka Panel Peneliti"):
            st.write("Panel ini digunakan untuk melihat data eksperimen.")
            if os.path.exists(DB_SQLITE_FILE):
                conn = sqlite3.connect(DB_SQLITE_FILE)
                import pandas as pd
                df = pd.read_sql_query("SELECT * FROM chat_logs", conn)
                st.dataframe(df)
                conn.close()
                
    elif kunci_admin != "":
        st.error("Akses ditolak. Sandi tidak valid.")
