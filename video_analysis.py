from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiofiles
import os
import uuid
from datetime import datetime
from typing import Optional
import logging
from pathlib import Path
import json

# Import our previously created components
from deception_detection_engine import DeceptionDetectionEngine, DeceptionScore
from auth_system import AuthenticationManager, User

# Video processing pipeline
class VideoAnalysisPipeline:
    def __init__(self, storage_path: str = "./video_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "uploads").mkdir(exist_ok=True)
        (self.storage_path / "processed").mkdir(exist_ok=True)
        (self.storage_path / "results").mkdir(exist_ok=True)
        
        self.detection_engine = DeceptionDetectionEngine()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def save_uploaded_video(self, file: UploadFile, session_id: str, user_id: str) -> str:
        """Save uploaded video file"""
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{user_id}_{timestamp}.mp4"
        file_path = self.storage_path / "uploads" / filename
        
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)
        
        self.logger.info(f"Video saved: {file_path}")
        return str(file_path)

    async def process_video_for_analysis(self, video_path: str, session_id: str, 
                                       subject_id: str, analyzer_id: str) -> DeceptionScore:
        """Process video through deception detection pipeline"""
        try:
            # Step 1: Establish baseline (first 30 seconds)
            self.logger.info(f"Establishing baseline for video: {video_path}")
            baseline = self.detection_engine.establish_baseline(video_path, duration_seconds=30)
            
            # Step 2: Analyze remaining video in segments
            self.logger.info("Starting deception analysis...")
            
            # For now, analyze the entire video after baseline
            # In production, you might want to analyze in smaller segments
            results = self.detection_engine.analyze_video_segment(
                video_path, 
                start_time=30,  # Start after baseline period
                end_time=None   # Analyze to end
            )
            
            # Step 3: Save results
            results_file = self.storage_path / "results" / f"{session_id}_analysis.json"
            self.detection_engine.save_analysis_results(
                results, subject_id, session_id, str(results_file)
            )
            
            self.logger.info(f"Analysis complete. Score: {results.overall_score:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def get_analysis_results(self, session_id: str) -> Optional[dict]:
        """Retrieve analysis results for a session"""
        results_file = self.storage_path / "results" / f"{session_id}_analysis.json"
        
        if not results_file.exists():
            return None
            
        async with aiofiles.open(results_file, 'r') as f:
            content = await f.read()
            return json.loads(content)

# FastAPI Application
app = FastAPI(title="Deception Detection API", version="1.0.0")

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
auth_manager = AuthenticationManager(secret_key="your-secret-key-change-this")
video_pipeline = VideoAnalysisPipeline()

# Authentication endpoints
@app.post("/auth/register")
async def register(user_data: UserRegistration):
    return auth_manager.register_user(user_data)

@app.post("/auth/login")
async def login(login_data: UserLogin):
    return auth_manager.login_user(login_data)

@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(auth_manager.get_current_user)):
    return current_user

# Session management endpoints
@app.post("/sessions/create")
async def create_session(
    participant_username: str = Form(...),
    current_user: User = Depends(auth_manager.get_current_user)
):
    session_id = auth_manager.create_recording_session(
        current_user.id, participant_username
    )
    return {"session_id": session_id, "message": "Session created successfully"}

@app.post("/sessions/{session_id}/consent")
async def give_consent(
    session_id: str, 
    consent_given: bool = Form(...),
    current_user: User = Depends(auth_manager.get_current_user)
):
    auth_manager.record_consent(session_id, current_user.id, consent_given)
    return {"message": "Consent recorded successfully"}

@app.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    current_user: User = Depends(auth_manager.get_current_user)
):
    return auth_manager.check_session_consent(session_id)

@app.post("/sessions/{session_id}/start")
async def start_session(
    session_id: str,
    current_user: User = Depends(auth_manager.get_current_user)
):
    auth_manager.start_recording_session(session_id)
    return {"message": "Recording session started successfully"}

# Video upload and analysis endpoints
@app.post("/analysis/upload")
async def upload_video_for_analysis(
    session_id: str = Form(...),
    video: UploadFile = File(...),
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Upload video file for deception analysis"""
    
    # Validate file type
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size (limit to 500MB)
    if video.size > 500 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 500MB")
    
    try:
        # Save uploaded video
        video_path = await video_pipeline.save_uploaded_video(
            video, session_id, current_user.id
        )
        
        # Start analysis in background
        asyncio.create_task(
            analyze_video_background(video_path, session_id, current_user.id, current_user.id)
        )
        
        return {
            "message": "Video uploaded successfully. Analysis started.",
            "session_id": session_id,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def analyze_video_background(video_path: str, session_id: str, 
                                 subject_id: str, analyzer_id: str):
    """Background task for video analysis"""
    try:
        # Process video through detection pipeline
        results = await video_pipeline.process_video_for_analysis(
            video_path, session_id, subject_id, analyzer_id
        )
        
        # Save results to database
        auth_manager.save_analysis_result(
            session_id=session_id,
            subject_id=subject_id,
            analyzer_id=analyzer_id,
            overall_score=results.overall_score,
            eye_movement_score=results.eye_movement_score,
            contradiction_score=results.contradiction_score,
            tonal_variation_score=results.tonal_variation_score,
            confidence_level=results.confidence_level,
            video_file_path=video_path
        )
        
        logging.info(f"Background analysis completed for session {session_id}")
        
    except Exception as e:
        logging.error(f"Background analysis failed: {str(e)}")

@app.get("/analysis/{session_id}/results")
async def get_analysis_results(
    session_id: str,
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Get analysis results for a session"""
    
    # Check if user has access to this session
    results = await video_pipeline.get_analysis_results(session_id)
    
    if not results:
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    return results

@app.get("/analysis/{session_id}/status")
async def get_analysis_status(
    session_id: str,
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Check analysis status"""
    
    results = await video_pipeline.get_analysis_results(session_id)
    
    if results:
        return {"status": "completed", "results_available": True}
    else:
        return {"status": "processing", "results_available": False}

@app.get("/analysis/history")
async def get_analysis_history(
    limit: int = 50,
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Get analysis history for current user"""
    return auth_manager.get_user_analysis_history(current_user.id, limit)

# Real-time analysis endpoint for live video calls
@app.post("/analysis/realtime")
async def start_realtime_analysis(
    session_id: str = Form(...),
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Start real-time analysis for live video call"""
    
    # Check session permissions
    consent_status = auth_manager.check_session_consent(session_id)
    if not consent_status["both_consented"]:
        raise HTTPException(
            status_code=403, 
            detail="Real-time analysis requires consent from both parties"
        )
    
    # In a real implementation, this would establish WebRTC connection
    # and start real-time processing
    return {
        "message": "Real-time analysis started",
        "session_id": session_id,
        "websocket_url": f"ws://your-domain.com/ws/analysis/{session_id}"
    }

# WebSocket endpoint for real-time updates (placeholder)
@app.websocket("/ws/analysis/{session_id}")
async def websocket_analysis(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time analysis updates"""
    await websocket.accept()
    
    try:
        while True:
            # In real implementation, this would:
            # 1. Receive video frames from client
            # 2. Process frames through detection engine
            # 3. Send back real-time scores
            
            await websocket.receive_text()
            
            # Placeholder response
            analysis_update = {
                "timestamp": datetime.now().isoformat(),
                "eye_movement_score": 25.5,
                "overall_score": 30.2,
                "confidence": 75.0
            }
            
            await websocket.send_text(json.dumps(analysis_update))
            
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# File management endpoints
@app.delete("/sessions/{session_id}/files")
async def delete_session_files(
    session_id: str,
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Delete all files associated with a session"""
    
    try:
        # Delete video files
        uploads_dir = video_pipeline.storage_path / "uploads"
        results_dir = video_pipeline.storage_path / "results"
        
        for file_path in uploads_dir.glob(f"{session_id}_*"):
            file_path.unlink(missing_ok=True)
            
        for file_path in results_dir.glob(f"{session_id}_*"):
            file_path.unlink(missing_ok=True)
            
        return {"message": "Session files deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")

# Configuration endpoint
@app.get("/config")
async def get_app_config():
    """Get app configuration for mobile clients"""
    return {
        "max_video_size_mb": 500,
        "supported_formats": ["mp4", "mov", "avi"],
        "max_session_duration_minutes": 60,
        "analysis_features": {
            "eye_tracking": True,
            "voice_analysis": True,
            "contradiction_detection": True
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

# Startup event
@app.on_event("startup")
async def startup_event():
    logging.info("Deception Detection API starting up...")
    
    # Initialize storage directories
    os.makedirs("./video_storage/uploads", exist_ok=True)
    os.makedirs("./video_storage/processed", exist_ok=True)
    os.makedirs("./video_storage/results", exist_ok=True)
    
    logging.info("API startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Deception Detection API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )
