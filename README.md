# AI CCTV Detection System 🎥🤖

Sistem CCTV Cerdas berbasis AI (YOLOv8 + ByteTrack + Rule Engine) yang mendeteksi pergerakan, melacak orang (persons), menganalisis perilaku mencurigakan berdasarkan rule yang dapat dikonfigurasi, dan mengirimkan notifikasi beserta foto snapshot langsung ke Telegram.

## Fitur Utama ✨
- **Multitasking & Multi-Camera:** Mendukung banyak kamera RTSP/IP secara bersamaan dengan pemrosesan multithreading.
- **Adaptive Detection:**
  - *Motion Gate:* Menggunakan MOG2 background subtraction untuk menghemat CPU/GPU saat tidak ada gerakan.
  - *Person Tracking:* YOLOv8 + ByteTrack untuk deteksi orang dan pelacakan lintasan.
- **Advanced Rule Engine (10+ Rules):** Sistem scoring (0-100) untuk perilaku mencurigakan seperti:
  - Masuk ke zona terlarang (Restricted Zone).
  - Berkeliaran terlalu lama (Loitering).
  - Pergerakan mendekat dengan cepat.
  - Aktivitas malam hari.
  - Mondar-mandir.
  - Dan lain-lain.
- **Notifikasi Telegram Cerdas:**
  - Mengirim alert awal saat terdeteksi aktivitas mencurigakan.
  - Mengirim snapshot periodik (setiap 2 menit) selama aktivitas masih berlangsung.
  - Anti-spam & Cooldown.
- **Database Terintegrasi:** Menyimpan log event, track history, dan info snapshot di SQLite menggunakan akses asinkron (aiosqlite).
- **Device-Agnostic:** Bisa berjalan di CPU, CUDA (GPU Nvidia), secara otomatis (Auto).

## Prasyarat 🛠️
- Python 3.11+ (atau Docker)
- Bot Telegram Token & Chat ID
- (Opsional) NVIDIA GPU & CUDA toolkit untuk performa maksimal.

## Instalasi & Menjalankan (Tanpa Docker) 🚀

1. **Clone & Masuk ke Folder**
   ```bash
   cd "CCTV AI DETECTION"
   ```

2. **Buat Virtual Environment (Opsional tapi disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi**
   - Copy `.env.example` ke `.env` dan isi `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID`.
   - Sesuaikan `config/config.yaml` untuk URL RTSP kamera Anda, area ROI, sensitivitas, dll.
   - Sesuaikan `config/rules.yaml` jika ingin mengubah skor atau parameter rule.

5. **Jalankan Aplikasi**
   ```bash
   python src/main.py
   ```

## Menjalankan dengan Docker 🐳

1. Pastikan `.env`, `config/config.yaml`, dan `config/rules.yaml` sudah disesuaikan.
2. Build & jalankan via Docker Compose:
   ```bash
   docker-compose up -d --build
   ```
3. Lihat log:
   ```bash
   docker-compose logs -f
   ```

## Pengaturan ROI (Region of Interest) 🎯
Untuk mengatur koordinat zona seperti `restricted_zone` atau `door_zone` pada `config.yaml`, Anda bisa menggunakan [Roboflow PolygonZone](https://roboflow.github.io/polygonzone/) untuk menggambar poligon pada sebuah frame gambar kamera Anda dan mendapatkan koordinat pixel-nya.

## Struktur Database 🗄️
Database SQLite tersimpan di `data/cctv_events.db` (dibuat otomatis).
Tabel yang tersedia:
- `cameras`: Daftar kamera.
- `events`: Log aktivitas mencurigakan (start, resolved, score, trigger rules).
- `snapshots`: Log foto yang diambil dan dikirim ke telegram.
- `tracks`: (Berisi koordinat lintasan per object/orang - otomatis di-flush berkala).
- `rule_triggers`: Alasan spesifik dari rule engine saat mendeteksi anomali.

## Troublehsooting ❓
- **Stream Lag/Delay:** Pastikan RTSP menggunakan koneksi stabil. System ini menggunakan *frame dropping* (selalu membaca frame terbaru) untuk meminimalisir lag RTSP bawaan OpenCV.
- **Telegram tidak mengirim pesan:** Pastikan Token Bot dan Chat ID benar. Jika pesan tertahan, sistem mengimplementasikan *RetryBackoff* dan penanganan *RateLimit* dari Telegram API.
