# Project-ALICE
ALICE: Advanced Logic Integrity &amp; Consistency Examiner


Prerequisites
Backend Requirements

Python 3.8+
PostgreSQL 12+
FFmpeg
OpenCV
4GB+ RAM
50GB+ storage

Mobile Development Requirements

Node.js 16+
React Native CLI
Android Studio (for Android)
Xcode (for iOS)

Backend Setup
1. Install Python Dependencies
bash# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary
pip install opencv-python mediapipe librosa speechrecognition
pip install python-multipart aiofiles asyncio-mqtt
pip install python-jose[cryptography] passlib[bcrypt]
pip install numpy pandas scikit-learn

# Install additional ML libraries
pip install torch torchvision torchaudio  # For advanced NLP
pip install transformers  # For contradiction detection
2. System Dependencies
Ubuntu/Debian:
bashsudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo apt install -y ffmpeg
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y libgl1-mesa-glx libglib2.0-0
macOS:
bashbrew install postgresql
brew install ffmpeg
brew install portaudio
3. Database Setup
sql-- Connect to PostgreSQL
sudo -u postgres psql

-- Create database and user
CREATE DATABASE deception_detection;
CREATE USER dduser WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE deception_detection TO dduser;
\q
4. Environment Configuration
Create .env file:
env# Database
DATABASE_URL=postgresql://dduser:your_secure_password@localhost/deception_detection

# Security
SECRET_KEY=your-very-secure-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
VIDEO_STORAGE_PATH=./video_storage
MAX_VIDEO_SIZE_MB=500

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# External Services (Optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your_s3_bucket
5. Run Backend Server
bash# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production with Gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Mobile App Setup
1. Initialize React Native Project
bash# Install React Native CLI
npm install -g react-native-cli

# Create new project
npx react-native init DeceptionDetectionApp
cd DeceptionDetectionApp

# Install dependencies
npm install @react-navigation/native @react-navigation/stack @react-navigation/bottom-tabs
npm install react-native-screens react-native-safe-area-context
npm install @react-native-async-storage/async-storage
npm install react-native-camera
npm install react-native-vector-icons
npm install react-native-permissions
2. Android Configuration
android/app/src/main/AndroidManifest.xml:
xml<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
android/app/build.gradle:
gradleandroid {
    compileSdkVersion 33
    
    defaultConfig {
        minSdkVersion 21
        targetSdkVersion 33
    }
}

dependencies {
    implementation project(':react-native-camera')
    implementation 'androidx.appcompat:appcompat:1.4.0'
}
3. Update App Configuration
config/api.js:
javascriptconst API_CONFIG = {
  BASE_URL: __DEV__ 
    ? 'http://10.0.2.2:8000'  // Android emulator
    : 'https://your-production-api.com',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3
};

export default API_CONFIG;
4. Build and Run
bash# Android
npx react-native run-android

# iOS (macOS only)
cd ios && pod install && cd ..
npx react-native run-ios
Production Deployment
1. Backend Deployment (Docker)
Dockerfile:
dockerfileFROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create storage directories
RUN mkdir -p video_storage/uploads video_storage/processed video_storage/results

EXPOSE 8000

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
docker-compose.yml:
yamlversion: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://dduser:password@db:5432/deception_detection
    depends_on:
      - db
    volumes:
      - ./video_storage:/app/video_storage
    
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: deception_detection
      POSTGRES_USER: dduser
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  postgres_data:
2. Mobile App Production Build
Android APK:
bash# Generate signed APK
cd android
./gradlew assembleRelease

# Output: android/app/build/outputs/apk/release/app-release.apk
iOS App Store:
bash# Build for release (Xcode required)
npx react-native run-ios --configuration Release
