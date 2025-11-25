"""
Feature Extractor for Mouse Movement Data
Converts raw mouse trajectories into feature vectors for ML
"""

import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import List, Dict, Tuple

class MouseFeatureExtractor:
    """Extract behavioral features from mouse movement trajectories"""
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of all feature names"""
        # Statistical features (8)
        self.feature_names.extend([
            'mean_velocity', 'std_velocity',
            'mean_acceleration', 'std_acceleration',
            'mean_angle_change', 'std_angle_change',
            'total_distance', 'straight_distance'
        ])
        
        # Behavioral features (8)
        self.feature_names.extend([
            'num_pauses', 'avg_pause_duration',
            'num_direction_changes', 'curvature',
            'path_efficiency', 'movement_smoothness',
            'click_count', 'avg_click_dwell'
        ])
        
        # Frequency features (8)
        self.feature_names.extend([
            'fft_vel_1', 'fft_vel_2', 'fft_vel_3', 'fft_vel_4',
            'fft_acc_1', 'fft_acc_2', 'dominant_freq', 'spectral_entropy'
        ])
    
    def extract_features(self, trajectory: List[Dict]) -> np.ndarray:
        """
        Extract all features from a mouse trajectory
        
        Args:
            trajectory: List of events [{x, y, timestamp, type}, ...]
        
        Returns:
            Feature vector of shape (24,)
        """
        if len(trajectory) < 5:
            return np.zeros(24)
        
        features = []
        
        # Extract coordinates and times
        xs = np.array([e['x'] for e in trajectory if e['type'] == 'move'])
        ys = np.array([e['y'] for e in trajectory if e['type'] == 'move'])
        ts = np.array([e['timestamp'] for e in trajectory if e['type'] == 'move'])
        
        if len(xs) < 3:
            return np.zeros(24)
        
        # Calculate derived values
        velocities = self._calculate_velocities(xs, ys, ts)
        accelerations = self._calculate_accelerations(velocities, ts)
        angles = self._calculate_angles(xs, ys)
        
        # === STATISTICAL FEATURES ===
        features.extend([
            np.mean(velocities),
            np.std(velocities),
            np.mean(np.abs(accelerations)),
            np.std(accelerations),
            np.mean(np.abs(angles)),
            np.std(angles),
            self._total_distance(xs, ys),
            self._straight_distance(xs, ys)
        ])
        
        # === BEHAVIORAL FEATURES ===
        features.extend([
            self._count_pauses(velocities),
            self._avg_pause_duration(velocities, ts),
            self._count_direction_changes(angles),
            self._calculate_curvature(xs, ys),
            self._path_efficiency(xs, ys),
            self._movement_smoothness(accelerations),
            self._count_clicks(trajectory),
            self._avg_click_dwell(trajectory)
        ])
        
        # === FREQUENCY FEATURES ===
        fft_features = self._frequency_features(velocities, accelerations)
        features.extend(fft_features)
        
        return np.array(features, dtype=np.float32)
    
    # ========== HELPER METHODS ==========
    
    def _calculate_velocities(self, xs, ys, ts):
        """Calculate velocities between consecutive points"""
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        dts = np.diff(ts) / 1000.0  # Convert to seconds
        dts = np.where(dts == 0, 1e-6, dts)  # Avoid division by zero
        return dists / dts
    
    def _calculate_accelerations(self, velocities, ts):
        """Calculate accelerations"""
        if len(velocities) < 2:
            return np.array([0])
        dv = np.diff(velocities)
        dt = np.diff(ts[:-1]) / 1000.0
        dt = np.where(dt == 0, 1e-6, dt)
        return dv / dt
    
    def _calculate_angles(self, xs, ys):
        """Calculate angle changes between consecutive segments"""
        if len(xs) < 3:
            return np.array([0])
        
        angles = []
        for i in range(1, len(xs) - 1):
            v1 = np.array([xs[i] - xs[i-1], ys[i] - ys[i-1]])
            v2 = np.array([xs[i+1] - xs[i], ys[i+1] - ys[i]])
            
            # Handle zero vectors
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                angles.append(0)
                continue
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        return np.array(angles)
    
    def _total_distance(self, xs, ys):
        """Total path length"""
        return np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
    
    def _straight_distance(self, xs, ys):
        """Straight-line distance from start to end"""
        return np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    
    def _count_pauses(self, velocities, threshold=10):
        """Count number of pauses (velocity < threshold)"""
        return np.sum(velocities < threshold)
    
    def _avg_pause_duration(self, velocities, ts, threshold=10):
        """Average duration of pauses"""
        is_pause = velocities < threshold
        if not np.any(is_pause):
            return 0
        
        pause_starts = np.where(np.diff(is_pause.astype(int)) == 1)[0]
        pause_ends = np.where(np.diff(is_pause.astype(int)) == -1)[0]
        
        if len(pause_starts) == 0 or len(pause_ends) == 0:
            return 0
        
        # Match starts with ends
        durations = []
        for start in pause_starts:
            matching_ends = pause_ends[pause_ends > start]
            if len(matching_ends) > 0:
                end = matching_ends[0]
                durations.append(ts[end] - ts[start])
        
        return np.mean(durations) if len(durations) > 0 else 0
    
    def _count_direction_changes(self, angles, threshold=np.pi/4):
        """Count significant direction changes"""
        return np.sum(np.abs(angles) > threshold)
    
    def _calculate_curvature(self, xs, ys):
        """Average curvature of path"""
        if len(xs) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(xs) - 1):
            # Three consecutive points
            p1 = np.array([xs[i-1], ys[i-1]])
            p2 = np.array([xs[i], ys[i]])
            p3 = np.array([xs[i+1], ys[i+1]])
            
            # Menger curvature formula
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a == 0 or b == 0 or c == 0:
                continue
            
            # Area of triangle using Heron's formula
            s = (a + b + c) / 2
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            
            # Curvature = 4 * Area / (a * b * c)
            curvature = 4 * area / (a * b * c) if (a * b * c) > 0 else 0
            curvatures.append(curvature)
        
        return np.mean(curvatures) if len(curvatures) > 0 else 0
    
    def _path_efficiency(self, xs, ys):
        """Path efficiency (straight distance / total distance)"""
        total = self._total_distance(xs, ys)
        straight = self._straight_distance(xs, ys)
        return straight / total if total > 0 else 0
    
    def _movement_smoothness(self, accelerations):
        """Smoothness of movement (inverse of jerk)"""
        if len(accelerations) < 2:
            return 0
        jerk = np.diff(accelerations)
        return 1.0 / (1.0 + np.std(jerk))
    
    def _count_clicks(self, trajectory):
        """Count number of clicks"""
        return sum(1 for e in trajectory if e['type'] == 'down')
    
    def _avg_click_dwell(self, trajectory):
        """Average click dwell time"""
        downs = [e for e in trajectory if e['type'] == 'down']
        ups = [e for e in trajectory if e['type'] == 'up']
        
        if len(downs) == 0 or len(ups) == 0:
            return 0
        
        dwells = []
        for down in downs:
            # Find corresponding up event
            matching_ups = [u for u in ups if u['timestamp'] > down['timestamp']]
            if len(matching_ups) > 0:
                dwells.append(matching_ups[0]['timestamp'] - down['timestamp'])
        
        return np.mean(dwells) if len(dwells) > 0 else 0
    
    def _frequency_features(self, velocities, accelerations):
        """Extract frequency domain features"""
        features = []
        
        # FFT of velocities (first 4 coefficients)
        if len(velocities) >= 8:
            fft_vel = np.abs(fft(velocities))[:len(velocities)//2]
            # Take 4 most significant coefficients
            sorted_vel = np.sort(fft_vel)[::-1]
            features.extend(sorted_vel[:4].tolist())
        else:
            features.extend([0, 0, 0, 0])
        
        # FFT of accelerations (first 2 coefficients)
        if len(accelerations) >= 8:
            fft_acc = np.abs(fft(accelerations))[:len(accelerations)//2]
            sorted_acc = np.sort(fft_acc)[::-1]
            features.extend(sorted_acc[:2].tolist())
        else:
            features.extend([0, 0])
        
        # Dominant frequency
        if len(velocities) >= 8:
            fft_vel = np.abs(fft(velocities))[:len(velocities)//2]
            dominant_freq = np.argmax(fft_vel)
            features.append(dominant_freq)
        else:
            features.append(0)
        
        # Spectral entropy
        if len(velocities) >= 8:
            fft_vel = np.abs(fft(velocities))[:len(velocities)//2]
            # Normalize to probability distribution
            power = fft_vel / np.sum(fft_vel) if np.sum(fft_vel) > 0 else fft_vel
            # Calculate entropy
            entropy = -np.sum(power * np.log2(power + 1e-10))
            features.append(entropy)
        else:
            features.append(0)
        
        return features


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    # Example trajectory data
    sample_trajectory = [
        {'x': 100, 'y': 200, 'timestamp': 0, 'type': 'move'},
        {'x': 105, 'y': 205, 'timestamp': 16, 'type': 'move'},
        {'x': 110, 'y': 210, 'timestamp': 32, 'type': 'move'},
        {'x': 115, 'y': 215, 'timestamp': 48, 'type': 'down'},
        {'x': 115, 'y': 215, 'timestamp': 120, 'type': 'up'},
        {'x': 120, 'y': 220, 'timestamp': 136, 'type': 'move'},
    ]
    
    extractor = MouseFeatureExtractor()
    features = extractor.extract_features(sample_trajectory)
    
    print("Extracted Features:")
    for name, value in zip(extractor.feature_names, features):
        print(f"  {name:25s}: {value:.4f}")
    
    print(f"\nTotal features: {len(features)}")