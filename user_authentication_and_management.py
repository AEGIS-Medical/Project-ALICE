from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import jwt
import bcrypt
import sqlite3
import uuid
from contextlib import contextmanager
import logging

# Models
class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class RecordingConsent(BaseModel):
    session_id: str
    participant_id: str
    consent_given: bool
    timestamp: datetime

class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    created_at: datetime
    is_active: bool

class AuthenticationManager:
    def __init__(self, secret_key: str, database_path: str = "users.db"):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.database_path = database_path
        self.security = HTTPBearer()
        
        # Initialize database
        self._init_database()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Recording sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recording_sessions (
                    id TEXT PRIMARY KEY,
                    initiator_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    FOREIGN KEY (initiator_id) REFERENCES users (id),
                    FOREIGN KEY (participant_id) REFERENCES users (id)
                )
            """)
            
            # Recording consent table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recording_consent (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    consent_given BOOLEAN NOT NULL,
                    consent_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (session_id) REFERENCES recording_sessions (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    analyzer_id TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    eye_movement_score REAL NOT NULL,
                    contradiction_score REAL NOT NULL,
                    tonal_variation_score REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    video_file_path TEXT,
                    FOREIGN KEY (session_id) REFERENCES recording_sessions (id),
                    FOREIGN KEY (subject_id) REFERENCES users (id),
                    FOREIGN KEY (analyzer_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def register_user(self, user_data: UserRegistration) -> Dict:
        """Register a new user"""
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(user_data.password)
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (id, username, email, full_name, password_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, user_data.username, user_data.email, 
                      user_data.full_name, password_hash))
                conn.commit()
                
                self.logger.info(f"User registered: {user_data.username}")
                return {"user_id": user_id, "message": "User registered successfully"}
                
        except sqlite3.IntegrityError as e:
            if "username" in str(e):
                raise HTTPException(status_code=400, detail="Username already exists")
            elif "email" in str(e):
                raise HTTPException(status_code=400, detail="Email already exists")
            else:
                raise HTTPException(status_code=400, detail="Registration failed")

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, email, full_name, password_hash, created_at, is_active
                FROM users WHERE username = ? AND is_active = TRUE
            """, (username,))
            
            user_data = cursor.fetchone()
            if not user_data:
                return None
            
            if not self.verify_password(password, user_data['password_hash']):
                return None
            
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                full_name=user_data['full_name'],
                created_at=datetime.fromisoformat(user_data['created_at']),
                is_active=user_data['is_active']
            )

    def login_user(self, login_data: UserLogin) -> Dict:
        """Login user and return access token"""
        user = self.authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={"sub": user.username, "user_id": user.id}, 
            expires_delta=access_token_expires
        )
        
        self.logger.info(f"User logged in: {user.username}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
            "user": user.dict()
        }

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current authenticated user"""
        token = credentials.credentials
        payload = self.verify_token(token)
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, email, full_name, created_at, is_active
                FROM users WHERE id = ? AND is_active = TRUE
            """, (user_id,))
            
            user_data = cursor.fetchone()
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                full_name=user_data['full_name'],
                created_at=datetime.fromisoformat(user_data['created_at']),
                is_active=user_data['is_active']
            )

    def create_recording_session(self, initiator_id: str, participant_username: str, 
                               session_type: str = "video_call") -> str:
        """Create a new recording session"""
        # Get participant ID
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (participant_username,))
            participant_data = cursor.fetchone()
            
            if not participant_data:
                raise HTTPException(status_code=404, detail="Participant not found")
            
            participant_id = participant_data['id']
            session_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO recording_sessions (id, initiator_id, participant_id, session_type)
                VALUES (?, ?, ?, ?)
            """, (session_id, initiator_id, participant_id, session_type))
            conn.commit()
            
            self.logger.info(f"Recording session created: {session_id}")
            return session_id

    def record_consent(self, session_id: str, user_id: str, consent_given: bool, 
                      ip_address: str = None, user_agent: str = None) -> bool:
        """Record user consent for recording"""
        consent_id = str(uuid.uuid4())
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recording_consent 
                (id, session_id, user_id, consent_given, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (consent_id, session_id, user_id, consent_given, ip_address, user_agent))
            conn.commit()
            
            self.logger.info(f"Consent recorded for session {session_id}: {consent_given}")
            return True

    def check_session_consent(self, session_id: str) -> Dict:
        """Check if both participants have given consent"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT initiator_id, participant_id FROM recording_sessions 
                WHERE id = ?
            """, (session_id,))
            session_data = cursor.fetchone()
            
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Check consent for both users
            cursor.execute("""
                SELECT user_id, consent_given FROM recording_consent 
                WHERE session_id = ? ORDER BY consent_timestamp DESC
            """, (session_id,))
            
            consents = cursor.fetchall()
            consent_status = {}
            
            for consent in consents:
                if consent['user_id'] not in consent_status:
                    consent_status[consent['user_id']] = consent['consent_given']
            
            initiator_consent = consent_status.get(session_data['initiator_id'], False)
            participant_consent = consent_status.get(session_data['participant_id'], False)
            
            return {
                "session_id": session_id,
                "initiator_consent": initiator_consent,
                "participant_consent": participant_consent,
                "both_consented": initiator_consent and participant_consent
            }

    def start_recording_session(self, session_id: str) -> bool:
        """Start recording session if both parties have consented"""
        consent_status = self.check_session_consent(session_id)
        
        if not consent_status["both_consented"]:
            raise HTTPException(
                status_code=403, 
                detail="Cannot start recording without consent from both parties"
            )
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE recording_sessions 
                SET status = 'active', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (session_id,))
            conn.commit()
            
            self.logger.info(f"Recording session started: {session_id}")
            return True

    def save_analysis_result(self, session_id: str, subject_id: str, analyzer_id: str,
                           overall_score: float, eye_movement_score: float,
                           contradiction_score: float, tonal_variation_score: float,
                           confidence_level: float, video_file_path: str = None):
        """Save deception analysis results"""
        result_id = str(uuid.uuid4())
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_results 
                (id, session_id, subject_id, analyzer_id, overall_score, 
                 eye_movement_score, contradiction_score, tonal_variation_score,
                 confidence_level, video_file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (result_id, session_id, subject_id, analyzer_id, overall_score,
                  eye_movement_score, contradiction_score, tonal_variation_score,
                  confidence_level, video_file_path))
            conn.commit()
            
            self.logger.info(f"Analysis result saved: {result_id}")
            return result_id

    def get_user_analysis_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get analysis history for a user"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ar.*, rs.session_type, u.username as subject_username
                FROM analysis_results ar
                JOIN recording_sessions rs ON ar.session_id = rs.id
                JOIN users u ON ar.subject_id = u.id
                WHERE ar.analyzer_id = ? OR ar.subject_id = ?
                ORDER BY ar.analysis_timestamp DESC
                LIMIT ?
            """, (user_id, user_id, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]

# FastAPI integration
app = FastAPI(title="Deception Detection API")
auth_manager = AuthenticationManager(secret_key="your-secret-key-here")

@app.post("/auth/register")
async def register(user_data: UserRegistration):
    return auth_manager.register_user(user_data)

@app.post("/auth/login")
async def login(login_data: UserLogin):
    return auth_manager.login_user(login_data)

@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(auth_manager.get_current_user)):
    return current_user

@app.post("/sessions/create")
async def create_session(participant_username: str, 
                        current_user: User = Depends(auth_manager.get_current_user)):
    session_id = auth_manager.create_recording_session(
        current_user.id, participant_username
    )
    return {"session_id": session_id}

@app.post("/sessions/{session_id}/consent")
async def give_consent(session_id: str, consent_given: bool,
                      current_user: User = Depends(auth_manager.get_current_user)):
    auth_manager.record_consent(session_id, current_user.id, consent_given)
    return {"message": "Consent recorded"}

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str,
                           current_user: User = Depends(auth_manager.get_current_user)):
    return auth_manager.check_session_consent(session_id)

@app.post("/sessions/{session_id}/start")
async def start_session(session_id: str,
                       current_user: User = Depends(auth_manager.get_current_user)):
    auth_manager.start_recording_session(session_id)
    return {"message": "Recording session started"}

@app.get("/analysis/history")
async def get_analysis_history(current_user: User = Depends(auth_manager.get_current_user)):
    return auth_manager.get_user_analysis_history(current_user.id)
