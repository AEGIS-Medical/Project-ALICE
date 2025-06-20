import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import librosa
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class DeceptionScore:
    overall_score: float
    eye_movement_score: float
    contradiction_score: float
    tonal_variation_score: float
    confidence_level: float
    timestamp: datetime

class DeceptionDetectionEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        self.baseline_eye_position = None
        self.statement_history = []
        self.baseline_tone = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def establish_baseline(self, video_path: str, duration_seconds: int = 30) -> Dict:
        """
        Establish baseline measurements for eye position and vocal tone
        """
        cap = cv2.VideoCapture(video_path)
        
        eye_positions = []
        audio_features = []
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frames = int(fps * duration_seconds)
        
        while frame_count < target_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process eye tracking
            eye_center = self._get_eye_center(frame)
            if eye_center:
                eye_positions.append(eye_center)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate baseline eye position (average)
        if eye_positions:
            self.baseline_eye_position = {
                'x': np.mean([pos[0] for pos in eye_positions]),
                'y': np.mean([pos[1] for pos in eye_positions])
            }
        
        # Extract audio for baseline tone analysis
        audio_data, sample_rate = librosa.load(video_path)
        self.baseline_tone = self._extract_audio_features(audio_data, sample_rate)
        
        baseline_data = {
            'eye_position': self.baseline_eye_position,
            'tone_features': self.baseline_tone,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Baseline established: {baseline_data}")
        return baseline_data

    def _get_eye_center(self, frame) -> Optional[Tuple[float, float]]:
        """
        Extract eye center position from frame
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Get left and right eye centers
            left_eye_points = []
            right_eye_points = []
            
            for idx in self.LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                left_eye_points.append([landmark.x * w, landmark.y * h])
            
            for idx in self.RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                right_eye_points.append([landmark.x * w, landmark.y * h])
            
            # Calculate center points
            left_center = np.mean(left_eye_points, axis=0)
            right_center = np.mean(right_eye_points, axis=0)
            
            # Average of both eyes
            eye_center = ((left_center[0] + right_center[0]) / 2, 
                         (left_center[1] + right_center[1]) / 2)
            
            return eye_center
        
        return None

    def _extract_audio_features(self, audio_data, sample_rate) -> Dict:
        """
        Extract audio features for tone analysis
        """
        # Extract fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # Extract other features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        
        features = {
            'pitch_mean': np.mean(pitch_values) if pitch_values else 0,
            'pitch_std': np.std(pitch_values) if pitch_values else 0,
            'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
            'spectral_centroid_mean': np.mean(spectral_centroids)
        }
        
        return features

    def analyze_video_segment(self, video_path: str, start_time: float = 0, 
                            end_time: float = None) -> DeceptionScore:
        """
        Analyze a video segment for deception indicators
        """
        if not self.baseline_eye_position or not self.baseline_tone:
            raise ValueError("Baseline must be established first")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
        
        eye_deviations = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = frame_count / fps + start_time
            if end_time and current_time > end_time:
                break
            
            # Analyze eye movement
            eye_center = self._get_eye_center(frame)
            if eye_center:
                deviation = self._calculate_eye_deviation(eye_center)
                eye_deviations.append(deviation)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate eye movement score
        eye_score = self._calculate_eye_movement_score(eye_deviations)
        
        # Extract and analyze audio for this segment
        audio_data, sample_rate = librosa.load(video_path, 
                                             offset=start_time, 
                                             duration=end_time-start_time if end_time else None)
        
        current_tone = self._extract_audio_features(audio_data, sample_rate)
        tonal_score = self._calculate_tonal_variation_score(current_tone)
        
        # Transcribe speech for contradiction analysis
        speech_text = self._transcribe_audio_segment(audio_data, sample_rate)
        contradiction_score = self._analyze_contradictions(speech_text)
        
        # Calculate overall deception score
        overall_score = self._calculate_overall_score(eye_score, contradiction_score, tonal_score)
        
        return DeceptionScore(
            overall_score=overall_score,
            eye_movement_score=eye_score,
            contradiction_score=contradiction_score,
            tonal_variation_score=tonal_score,
            confidence_level=self._calculate_confidence(eye_deviations, speech_text),
            timestamp=datetime.now()
        )

    def _calculate_eye_deviation(self, current_position: Tuple[float, float]) -> float:
        """
        Calculate deviation from baseline eye position
        """
        baseline_x = self.baseline_eye_position['x']
        baseline_y = self.baseline_eye_position['y']
        
        deviation = np.sqrt((current_position[0] - baseline_x)**2 + 
                           (current_position[1] - baseline_y)**2)
        return deviation

    def _calculate_eye_movement_score(self, deviations: List[float]) -> float:
        """
        Calculate eye movement deception score (0-100)
        """
        if not deviations:
            return 0
        
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        
        # Normalize to 0-100 scale (higher = more suspicious)
        # This is a simplified scoring - would need calibration with real data
        score = min(100, (avg_deviation / 50) * 40 + (max_deviation / 100) * 60)
        return score

    def _calculate_tonal_variation_score(self, current_tone: Dict) -> float:
        """
        Calculate tonal variation deception score
        """
        if not self.baseline_tone:
            return 0
        
        pitch_diff = abs(current_tone['pitch_mean'] - self.baseline_tone['pitch_mean'])
        pitch_variation = abs(current_tone['pitch_std'] - self.baseline_tone['pitch_std'])
        
        # Normalize to 0-100 scale
        score = min(100, (pitch_diff / 50) * 30 + (pitch_variation / 20) * 70)
        return score

    def _transcribe_audio_segment(self, audio_data, sample_rate) -> str:
        """
        Transcribe audio segment to text
        """
        try:
            # Convert to format suitable for speech recognition
            import io
            import wave
            
            # Simple transcription - in production, use more robust service
            return "Sample transcribed text"  # Placeholder
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""

    def _analyze_contradictions(self, current_statement: str) -> float:
        """
        Analyze current statement against statement history for contradictions
        """
        if not current_statement:
            return 0
        
        # Add to statement history
        self.statement_history.append({
            'text': current_statement,
            'timestamp': datetime.now()
        })
        
        # Simple contradiction detection (would use NLP models in production)
        contradiction_indicators = 0
        
        # Look for contradictory keywords/phrases
        contradictory_pairs = [
            ('yes', 'no'), ('always', 'never'), ('did', "didn't"),
            ('was', "wasn't"), ('will', "won't")
        ]
        
        for statement in self.statement_history[-5:]:  # Check last 5 statements
            for pair in contradictory_pairs:
                if pair[0] in current_statement.lower() and pair[1] in statement['text'].lower():
                    contradiction_indicators += 1
                elif pair[1] in current_statement.lower() and pair[0] in statement['text'].lower():
                    contradiction_indicators += 1
        
        # Normalize to 0-100 scale
        score = min(100, contradiction_indicators * 25)
        return score

    def _calculate_overall_score(self, eye_score: float, contradiction_score: float, 
                               tonal_score: float) -> float:
        """
        Calculate weighted overall deception score
        """
        # Weighted average (weights can be adjusted based on research)
        weights = {'eye': 0.4, 'contradiction': 0.4, 'tonal': 0.2}
        
        overall = (eye_score * weights['eye'] + 
                  contradiction_score * weights['contradiction'] + 
                  tonal_score * weights['tonal'])
        
        return min(100, overall)

    def _calculate_confidence(self, eye_deviations: List[float], speech_text: str) -> float:
        """
        Calculate confidence level in the analysis
        """
        # Higher confidence with more data points and clearer signals
        data_quality = len(eye_deviations) / 100  # Normalize frame count
        speech_quality = len(speech_text) / 100 if speech_text else 0
        
        confidence = min(100, (data_quality + speech_quality) * 50)
        return confidence

    def save_analysis_results(self, results: DeceptionScore, subject_id: str, 
                            session_id: str, file_path: str):
        """
        Save analysis results to file
        """
        data = {
            'subject_id': subject_id,
            'session_id': session_id,
            'timestamp': results.timestamp.isoformat(),
            'scores': {
                'overall': results.overall_score,
                'eye_movement': results.eye_movement_score,
                'contradiction': results.contradiction_score,
                'tonal_variation': results.tonal_variation_score
            },
            'confidence_level': results.confidence_level
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Analysis results saved to {file_path}")

# Example usage
if __name__ == "__main__":
    detector = DeceptionDetectionEngine()
    
    # Establish baseline
    baseline = detector.establish_baseline("baseline_video.mp4", duration_seconds=30)
    
    # Analyze a segment
    results = detector.analyze_video_segment("test_video.mp4", start_time=0, end_time=60)
    
    print(f"Deception Score: {results.overall_score:.2f}/100")
    print(f"Confidence: {results.confidence_level:.2f}%")
    
    # Save results
    detector.save_analysis_results(results, "subject_001", "session_123", "analysis_results.json")
