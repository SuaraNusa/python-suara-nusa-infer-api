# Menggunakan Python sebagai base image
FROM python:3.9-slim

# Mengatur environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Menginstal dependencies sistem untuk FFmpeg dan kebutuhan lainnya
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Membuat direktori kerja untuk aplikasi
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Menginstal dependencies Python
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file aplikasi ke dalam container
COPY . /app/

# Menentukan port yang akan digunakan oleh aplikasi
EXPOSE $PORT

# Perintah untuk menjalankan aplikasi Flask
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
