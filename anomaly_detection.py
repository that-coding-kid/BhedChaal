import numpy as np
import cv2
import math
import json
import os
import time
from datetime import datetime

class AnomalyDetector:
    """
    Class for detecting anomalies in crowd movement patterns
    Current implementation focuses on detecting entities moving against the major flow direction
    """
    
    def __init__(self, angle_threshold=65, history_length=5, anomaly_persistence=15):
        """
        Initialize the anomaly detector
        
        Parameters:
        - angle_threshold: Minimum angle difference (in degrees) from majority flow to be considered an anomaly
        - history_length: Number of recent frames to consider for stable flow direction
        - anomaly_persistence: Number of frames an entity remains marked as anomaly after detection
        """
        self.angle_threshold = angle_threshold
        self.history_length = history_length
        self.anomaly_persistence = anomaly_persistence
        self.major_flow_history = []
        self.anomalies = set()  # Store IDs of currently active anomalies
        self.anomaly_timers = {}  # Track how long each anomaly should persist {track_id: frames_remaining}
        self.last_known_positions = {}  # Store the last known position of each anomaly
        
        # For tracking anomaly detection events
        self.anomaly_events = {}  # Store info about anomalies {track_id: {"first_detected": timestamp, "frame": frame_num}}
        self.current_frame = 0  # Current frame counter
        
        # Create directory for storing anomaly data
        self.anomaly_data_dir = os.path.join("video_data", "anomalies")
        if not os.path.exists(self.anomaly_data_dir):
            os.makedirs(self.anomaly_data_dir)
    
    def reset(self):
        """Reset the detector state"""
        self.major_flow_history = []
        self.anomalies = set()
        self.anomaly_timers = {}
        self.last_known_positions = {}
        self.anomaly_events = {}
        self.current_frame = 0
    
    def vector_angle(self, v1, v2):
        """
        Calculate the angle between two vectors in degrees
        
        Parameters:
        - v1, v2: Vectors as (x, y) tuples
        
        Returns:
        - Angle between vectors in degrees
        """
        # Check if either vector is (0,0)
        if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
            return 0
            
        # Calculate dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Calculate magnitudes
        v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Calculate cosine of angle
        cos_angle = dot_product / (v1_mag * v2_mag)
        
        # Handle floating point errors
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        # Calculate angle in degrees
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def detect_counter_flow(self, movement_vectors, include_track_ids=True, frame_number=None):
        """
        Detect entities moving against the major flow direction
        
        Parameters:
        - movement_vectors: List of (position, vector, color, [track_id]) tuples with movement data
        - include_track_ids: Boolean indicating if track_ids are included in movement_vectors
        - frame_number: Current frame number for logging anomaly events
        
        Returns:
        - counter_flow_ids: Set of track IDs moving against major flow
        - major_flow_vector: The calculated major flow direction (x, y)
        """
        # Update current frame number if provided
        if frame_number is not None:
            self.current_frame = frame_number
        
        # Get all current track IDs to help clean up expired timers
        current_track_ids = set()
        
        # First, update our record of the last known positions
        if include_track_ids:
            for pos, _, _, track_id in movement_vectors:
                current_track_ids.add(track_id)
                # Store last known position for every track
                self.last_known_positions[track_id] = pos
        
        # Process anomaly timers - decrease counters and remove expired ones
        # Anomalies will persist for self.anomaly_persistence frames (60 frames = ~2 seconds at 30fps)
        # even if the track is no longer present or is no longer moving against the flow
        for track_id in list(self.anomaly_timers.keys()):
            # Decrease timer
            self.anomaly_timers[track_id] -= 1
            
            # Remove expired timers only when the counter reaches zero
            # This ensures anomalies persist for their full duration even if tracks disappear
            if self.anomaly_timers[track_id] <= 0:
                del self.anomaly_timers[track_id]
                # Also remove from anomalies set if it was there
                if track_id in self.anomalies:
                    self.anomalies.remove(track_id)
                # Clean up last known position
                if track_id in self.last_known_positions:
                    del self.last_known_positions[track_id]
                
                # Record the end of the anomaly in events
                if track_id in self.anomaly_events:
                    self.anomaly_events[track_id]["end_frame"] = self.current_frame
                    self.save_anomaly_data(track_id)
        
        # Prepare to detect new anomalies for this frame
        new_anomalies = set()
        
        # Check if we have enough vectors for anomaly detection
        if len(movement_vectors) < 2:
            # Not enough vectors to establish a reliable pattern
            return self.anomalies, (0, 0)  # Return all anomalies (including persisting ones)
        
        # Filter out zero or very small movement vectors that might be noise
        significant_vectors = []
        for item in movement_vectors:
            vector = item[1]  # Extract vector component
            # Calculate vector magnitude
            magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
            # Only include vectors with significant movement
            if magnitude > 1.0:  # Minimum magnitude threshold
                significant_vectors.append(item)
        
        # If no significant vectors, return existing anomalies
        if len(significant_vectors) < 2:
            return self.anomalies, (0, 0)
        
        # Calculate the average flow vector from significant vectors
        if include_track_ids:
            # Format: (position, vector, color, track_id)
            avg_x = sum(v[0] for _, v, _, _ in significant_vectors) / len(significant_vectors)
            avg_y = sum(v[1] for _, v, _, _ in significant_vectors) / len(significant_vectors)
        else:
            # Format: (position, vector, color)
            avg_x = sum(v[0] for _, v, _ in significant_vectors) / len(significant_vectors)
            avg_y = sum(v[1] for _, v, _ in significant_vectors) / len(significant_vectors)
        
        # Add to history
        self.major_flow_history.append((avg_x, avg_y))
        
        # Keep history at specified length
        while len(self.major_flow_history) > self.history_length:
            self.major_flow_history.pop(0)
        
        # Calculate the stable major flow direction using history
        major_flow_x = sum(x for x, _ in self.major_flow_history) / len(self.major_flow_history)
        major_flow_y = sum(y for _, y in self.major_flow_history) / len(self.major_flow_history)
        major_flow_vector = (major_flow_x, major_flow_y)
        
        # Skip anomaly detection if major flow vector is too small (near zero)
        major_flow_magnitude = math.sqrt(major_flow_x**2 + major_flow_y**2)
        if major_flow_magnitude < 1.0:
            return self.anomalies, major_flow_vector
        
        # Check each movement vector to identify those going against the major flow
        for idx, (pos, vector, color, *track_info) in enumerate(movement_vectors):
            # Get track ID if provided
            track_id = track_info[0] if track_info else idx
            
            # Skip if this is already a known anomaly with an active timer
            if track_id in self.anomalies and track_id in self.anomaly_timers:
                continue
                
            # Calculate magnitude of this vector
            vector_magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
            
            # Skip vectors with too little movement
            if vector_magnitude < 1.0:
                continue
                
            # Calculate angle between this vector and major flow vector
            angle = self.vector_angle(vector, major_flow_vector)
            
            # If angle is greater than threshold, mark as an anomaly
            if angle > self.angle_threshold:
                new_anomalies.add(track_id)
                # Set or reset the persistence timer for this anomaly
                self.anomaly_timers[track_id] = self.anomaly_persistence
                
                # Record the anomaly event
                if track_id not in self.anomaly_events:
                    # Only record first detection for new anomalies
                    self.anomaly_events[track_id] = {
                        "id": track_id,
                        "first_detected": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "start_frame": self.current_frame,
                        "position": [float(pos[0]), float(pos[1])],
                        "vector": [float(vector[0]), float(vector[1])],
                        "angle": float(angle),
                        "major_flow": [float(major_flow_x), float(major_flow_y)]
                    }
        
        # Update the anomalies set with new anomalies
        self.anomalies.update(new_anomalies)
        
        return self.anomalies, major_flow_vector
    
    def save_anomaly_data(self, track_id):
        """Save anomaly data to JSON file"""
        if track_id not in self.anomaly_events:
            return
        
        # Generate a unique filename
        timestamp = self.anomaly_events[track_id]["first_detected"].replace(" ", "_").replace(":", "-")
        filename = f"anomaly_{track_id}_{timestamp}.json"
        filepath = os.path.join(self.anomaly_data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.anomaly_events[track_id], f, indent=2)
            
            # Remove from events after saving
            del self.anomaly_events[track_id]
        except Exception as e:
            print(f"Error saving anomaly data: {e}")
    
    def save_all_anomaly_data(self):
        """Save all pending anomaly data to JSON files"""
        # Add end frame to all active anomalies
        for track_id in list(self.anomaly_events.keys()):
            if "end_frame" not in self.anomaly_events[track_id]:
                self.anomaly_events[track_id]["end_frame"] = self.current_frame
            self.save_anomaly_data(track_id)
    
    def visualize_anomalies(self, frame, anomaly_positions, pulse_rate=5):
        """
        Visualize the detected anomalies on the frame
        
        Parameters:
        - frame: The frame to draw on
        - anomaly_positions: Dictionary mapping track_ids to positions (x, y)
        - pulse_rate: Rate of pulsating effect
        
        Returns:
        - Frame with visualized anomalies
        """
        # Make a copy of the frame to avoid modifying the original
        viz_frame = frame.copy()
        
        # Current time for pulsating effect
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Make a copy to avoid modifying the original
        all_anomaly_positions = anomaly_positions.copy() if anomaly_positions else {}
        
        # Add last known positions for anomalies that aren't in the current frame
        for track_id in self.anomalies:
            if track_id not in all_anomaly_positions and track_id in self.last_known_positions:
                all_anomaly_positions[track_id] = self.last_known_positions[track_id]
        
        # Draw markers for all anomalies with active timers (both current and persistent)
        for track_id in self.anomaly_timers.keys():
            # Skip if not in anomalies set (shouldn't happen but just in case)
            if track_id not in self.anomalies:
                continue
                
            # Skip if we don't have a position for this anomaly
            if track_id not in all_anomaly_positions:
                continue
                
            pos = all_anomaly_positions[track_id]
            
            # Create a pulsating effect
            pulse_value = int(127 + 127 * np.sin(current_time * pulse_rate))
            
            # Parameters for diamond shape - smaller size
            diamond_size = 8  # Reduced from 15
            diamond_points = np.array([
                [pos[0], pos[1] - diamond_size],
                [pos[0] + diamond_size, pos[1]],
                [pos[0], pos[1] + diamond_size],
                [pos[0] - diamond_size, pos[1]]
            ], np.int32)
            
            # Draw filled diamond with pulsating blue color
            cv2.fillPoly(viz_frame, [diamond_points], (pulse_value, 0, 0))  # Blue in BGR
            cv2.polylines(viz_frame, [diamond_points], True, (255, 0, 0), 2)  # Solid blue outline
            
            # Text labels and frame counter have been removed
        
        return viz_frame

    def draw_anomaly_markers(self, frame, anomaly_positions, pulse_rate=5):
        """
        Draw distinctive markers for anomalies on the frame
        
        Parameters:
        - frame: The frame to draw on (modified in-place)
        - anomaly_positions: Dictionary mapping track_ids to positions (x, y)
        - pulse_rate: Rate of pulsating effect
        """
        # Current time for pulsating effect
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Make a copy to avoid modifying the original
        all_anomaly_positions = anomaly_positions.copy() if anomaly_positions else {}
        
        # Add last known positions for anomalies that aren't in the current frame
        for track_id in self.anomalies:
            if track_id not in all_anomaly_positions and track_id in self.last_known_positions:
                all_anomaly_positions[track_id] = self.last_known_positions[track_id]
        
        # Draw markers for all anomalies with active timers (both current and persistent)
        for track_id in self.anomaly_timers.keys():
            # Skip if not in anomalies set (shouldn't happen but just in case)
            if track_id not in self.anomalies:
                continue
                
            # Skip if we don't have a position for this anomaly
            if track_id not in all_anomaly_positions:
                continue
                
            pos = all_anomaly_positions[track_id]
            
            # Create a pulsating effect
            pulse_value = int(127 + 127 * np.sin(current_time * pulse_rate))
            
            # Parameters for diamond shape - smaller size
            diamond_size = 8  # Reduced from 12
            diamond_points = np.array([
                [pos[0], pos[1] - diamond_size],
                [pos[0] + diamond_size, pos[1]],
                [pos[0], pos[1] + diamond_size],
                [pos[0] - diamond_size, pos[1]]
            ], np.int32)
            
            # Draw filled diamond with pulsating blue color
            cv2.fillPoly(frame, [diamond_points], (pulse_value, 0, 0))  # Blue in BGR
            cv2.polylines(frame, [diamond_points], True, (255, 0, 0), 2)  # Solid blue outline
            
            # Text labels and frame counter have been removed 