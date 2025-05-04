import numpy as np
import cv2
import math
import json
import os
import time
from datetime import datetime
from collections import defaultdict

class AnomalyDetector:
    """
    Class for detecting anomalies in crowd movement patterns
    Current implementation focuses on detecting entities moving against the major flow direction
    """
    
    def __init__(self, angle_threshold=65, history_length=5, anomaly_persistence=15, anomaly_threshold=30, stampede_threshold=35, max_bottlenecks=3):
        """
        Initialize the anomaly detector
        
        Parameters:
        - angle_threshold: Minimum angle difference (in degrees) from majority flow to be considered an anomaly
        - history_length: Number of recent frames to consider for stable flow direction
        - anomaly_persistence: Number of frames an entity remains marked as anomaly after detection
        - anomaly_threshold: Threshold to identify bottlenecks (when anomaly count exceeds this value)
        - stampede_threshold: Threshold to trigger stampede warning (when anomaly count exceeds this value)
        - max_bottlenecks: Maximum number of bottlenecks to identify (default: 3)
        """
        self.angle_threshold = angle_threshold
        self.history_length = history_length
        self.anomaly_persistence = anomaly_persistence
        self.anomaly_threshold = anomaly_threshold
        self.stampede_threshold = stampede_threshold
        self.max_bottlenecks = max_bottlenecks
        self.major_flow_history = []
        self.anomalies = set()  # Store IDs of currently active anomalies
        self.anomaly_timers = {}  # Track how long each anomaly should persist {track_id: frames_remaining}
        self.last_known_positions = {}  # Store the last known position of each anomaly
        
        # For tracking anomaly detection events
        self.anomaly_events = {}  # Store info about anomalies {track_id: {"first_detected": timestamp, "frame": frame_num}}
        self.current_frame = 0  # Current frame counter
        
        # For tracking bottleneck locations
        self.bottlenecks = []  # List of bottleneck locations [(x, y, width, height), ...]
        self.bottleneck_heatmap = np.zeros((600, 800), dtype=np.float32)  # Default size, will be resized as needed
        self.population_heatmap = None  # Will store the population density heatmap
        self.bottleneck_update_interval = 60  # Update bottleneck every 60 frames (2 seconds at 30fps)
        self.last_bottleneck_update = 0
        self.density_grid_size = 20  # Size of grid cells for density estimation
        self.last_anomaly_count = 0  # Track previous anomaly count for threshold/2 detection
        
        # Stampede detection
        self.stampede_warning = False  # Flag to indicate if stampede warning is active
        self.stampede_warning_started = None  # Frame when stampede warning started

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
        self.bottlenecks = []
        self.bottleneck_heatmap = np.zeros_like(self.bottleneck_heatmap)
        self.population_heatmap = None
        self.last_anomaly_count = 0
        self.stampede_warning = False
        self.stampede_warning_started = None
    
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
        for track_id in list(self.anomaly_timers.keys()):
            # Decrease timer
            self.anomaly_timers[track_id] -= 1
            
            # Remove expired timers only when the counter reaches zero
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
        
        # Update bottleneck heatmap with current anomaly positions
        self._update_bottleneck_heatmap()
        
        # Check if it's time to identify or remove bottlenecks
        current_anomaly_count = len(self.anomalies)
        
        # If anomaly count has dropped below threshold/2, clear all bottlenecks
        if len(self.bottlenecks) > 0 and current_anomaly_count < (self.anomaly_threshold / 2):
            print(f"Removing bottlenecks as anomaly count ({current_anomaly_count}) is below threshold/2 ({self.anomaly_threshold/2})")
            self.bottlenecks = []
        
        # Identify bottlenecks if we're above threshold and it's time to update
        elif (current_anomaly_count > self.anomaly_threshold and 
            (self.current_frame - self.last_bottleneck_update > self.bottleneck_update_interval or 
             len(self.bottlenecks) == 0)):
            self._identify_bottleneck()
            self.last_bottleneck_update = self.current_frame
        
        # Check for stampede warning condition
        if current_anomaly_count >= self.stampede_threshold:
            if not self.stampede_warning:
                self.stampede_warning = True
                self.stampede_warning_started = self.current_frame
                print(f"WARNING: Potential stampede detected! Anomaly count ({current_anomaly_count}) exceeds threshold ({self.stampede_threshold})")
        else:
            # Reset stampede warning when count drops below threshold
            if self.stampede_warning:
                print(f"Stampede warning cleared. Anomaly count ({current_anomaly_count}) now below threshold ({self.stampede_threshold})")
                self.stampede_warning = False
                self.stampede_warning_started = None
        
        # Store current anomaly count for next time
        self.last_anomaly_count = current_anomaly_count
        
        return self.anomalies, major_flow_vector
    
    def update_population_heatmap(self, density_map=None, person_positions=None):
        """
        Update the population density heatmap from density map and person positions
        
        Parameters:
        - density_map: Density map from CSRNet, warped to top view
        - person_positions: Dictionary of {id: position} of detected people
        """
        h, w = self.bottleneck_heatmap.shape
        
        # Initialize a new heatmap if needed
        if self.population_heatmap is None:
            self.population_heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Create temporary heatmap
        temp_heatmap = np.zeros_like(self.population_heatmap)
        
        # Add density map if available (with higher weight)
        if density_map is not None:
            # Make sure density map is same size as heatmap
            if density_map.shape != temp_heatmap.shape:
                try:
                    density_map = cv2.resize(density_map, (w, h))
                except:
                    # If resize fails, skip density map
                    pass
            
            # Add density map to temp heatmap
            if density_map.shape == temp_heatmap.shape:
                # Normalize density map
                density_norm = density_map / (np.max(density_map) + 1e-10)
                temp_heatmap += density_norm * 1.5  # Higher weight for density map
        
        # Add person positions if available
        if person_positions is not None and len(person_positions) > 0:
            for pos in person_positions.values():
                x, y = int(pos[0]), int(pos[1])
                # Skip positions outside the heatmap
                if not (0 <= x < w and 0 <= y < h):
                    continue
                    
                # Add a gaussian blob for each person
                sigma = 15
                x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
                gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
                temp_heatmap += gaussian
        
        # Normalize and add to cumulative heatmap with decay
        if np.max(temp_heatmap) > 0:
            temp_heatmap = temp_heatmap / np.max(temp_heatmap)
            # Apply decay to existing heatmap (0.95) and add new positions
            self.population_heatmap = self.population_heatmap * 0.95 + temp_heatmap * 0.05
    
    def _update_bottleneck_heatmap(self):
        """Update the heatmap used to identify bottlenecks based on anomalies"""
        # Skip if we don't have anomalies
        if not self.anomalies:
            return
            
        # Get all positions of current anomalies
        positions = []
        for track_id in self.anomalies:
            if track_id in self.last_known_positions:
                positions.append(self.last_known_positions[track_id])
                
        if not positions:
            return
            
        # Create a temporary heatmap for current positions
        h, w = self.bottleneck_heatmap.shape
        temp_heatmap = np.zeros_like(self.bottleneck_heatmap)
            
        # Add gaussian blobs at each anomaly position
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            # Skip positions outside the heatmap
            if not (0 <= x < w and 0 <= y < h):
                continue
                
            # Add a gaussian blob with sigma=15
            sigma = 15
            x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
            # Gaussian centered at the anomaly position
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            temp_heatmap += gaussian
            
        # Normalize and add to cumulative heatmap with decay
        if np.max(temp_heatmap) > 0:
            temp_heatmap = temp_heatmap / np.max(temp_heatmap)
            # Apply decay to existing heatmap (0.95) and add new positions
            self.bottleneck_heatmap = self.bottleneck_heatmap * 0.95 + temp_heatmap * 0.05
            
    def _identify_bottleneck(self):
        """
        Identify up to [max_bottlenecks] bottleneck regions as the most populous areas that also have anomalies.
        Uses a combination of population density and anomaly density heatmaps.
        """
        # If we don't have a population heatmap yet, fall back to anomaly-only method
        if self.population_heatmap is None or np.max(self.population_heatmap) <= 0:
            self._identify_bottleneck_anomaly_only()
            return
            
        # Get the dimensions
        h, w = self.bottleneck_heatmap.shape
        
        # Create a combined heatmap that prioritizes regions with both high population and anomalies
        # Higher weight (0.7) to population density, lower (0.3) to anomaly density
        combined_heatmap = self.population_heatmap * 0.7 + self.bottleneck_heatmap * 0.3
        
        # Only consider areas that have some anomalies
        min_anomaly_value = np.max(self.bottleneck_heatmap) * 0.3
        anomaly_mask = self.bottleneck_heatmap > min_anomaly_value
        
        # Apply the mask to the combined heatmap
        masked_heatmap = combined_heatmap.copy()
        masked_heatmap[~anomaly_mask] = 0
        
        # Skip if no valid regions
        if np.max(masked_heatmap) <= 0:
            self.bottlenecks = []
            return
            
        # Threshold the masked heatmap to focus on high-density regions
        threshold = np.max(masked_heatmap) * 0.6
        binary_mask = (masked_heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.bottlenecks = []
            return
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Keep only the max_bottlenecks largest contours
        contours = contours[:self.max_bottlenecks]
        
        # Process each contour to create a bottleneck
        self.bottlenecks = []
        for contour in contours:
            # Get bounding box
            x, y, width, height = cv2.boundingRect(contour)
            
            # Expand the box a bit
            expand_factor = 1.2
            center_x, center_y = x + width/2, y + height/2
            new_width = width * expand_factor
            new_height = height * expand_factor
            
            # Calculate new top-left corner
            new_x = max(0, int(center_x - new_width/2))
            new_y = max(0, int(center_y - new_height/2))
            
            # Ensure we don't exceed image boundaries
            new_width = min(int(new_width), w - new_x)
            new_height = min(int(new_height), h - new_y)
            
            # Add the bottleneck location
            self.bottlenecks.append((new_x, new_y, new_width, new_height))
        
        # Log the bottleneck detection
        print(f"Identified {len(self.bottlenecks)} bottlenecks at frame {self.current_frame} with {len(self.anomalies)} anomalies")
        for i, bottleneck in enumerate(self.bottlenecks):
            print(f"Bottleneck {i+1}: {bottleneck}")
    
    def _identify_bottleneck_anomaly_only(self):
        """Fallback method to identify bottlenecks based only on anomaly density"""
        if np.max(self.bottleneck_heatmap) <= 0:
            self.bottlenecks = []
            return
        
        # Find center of the highest density region
        h, w = self.bottleneck_heatmap.shape
        
        # Threshold the heatmap to focus on high-density regions
        threshold = np.max(self.bottleneck_heatmap) * 0.7
        mask = (self.bottleneck_heatmap > threshold).astype(np.uint8) * 255
        
        # Find contours in the thresholded heatmap
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.bottlenecks = []
            return
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Keep only the max_bottlenecks largest contours
        contours = contours[:self.max_bottlenecks]
        
        # Process each contour to create a bottleneck
        self.bottlenecks = []
        for contour in contours:
            # Get bounding box
            x, y, width, height = cv2.boundingRect(contour)
            
            # Expand the box a bit to ensure it includes nearby anomalies
            expand_factor = 1.2
            center_x, center_y = x + width/2, y + height/2
            new_width = width * expand_factor
            new_height = height * expand_factor
            
            # Calculate new top-left corner
            new_x = max(0, int(center_x - new_width/2))
            new_y = max(0, int(center_y - new_height/2))
            
            # Ensure we don't exceed image boundaries
            new_width = min(int(new_width), w - new_x)
            new_height = min(int(new_height), h - new_y)
            
            # Add the bottleneck location
            self.bottlenecks.append((new_x, new_y, new_width, new_height))
        
        # Log the bottleneck detection
        print(f"Identified {len(self.bottlenecks)} bottlenecks at frame {self.current_frame} based on anomaly density only, with {len(self.anomalies)} anomalies")
        for i, bottleneck in enumerate(self.bottlenecks):
            print(f"Bottleneck {i+1}: {bottleneck}")
    
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

    def project_bottleneck_to_original(self, inv_homography):
        """
        Project the bottleneck locations from top view to original view
        
        Parameters:
        - inv_homography: Inverse homography matrix to transform back to original view
        
        Returns:
        - List of (x, y, width, height) tuples with bottleneck locations in original view
          or None if no bottlenecks
        """
        if not self.bottlenecks:
            return None
            
        original_bottlenecks = []
        
        for bottleneck in self.bottlenecks:
            x, y, width, height = bottleneck
            
            # Get the corners of the bottleneck box in top view
            corners = np.array([
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height]
            ], dtype=np.float32)
            
            # Transform corners to original view
            original_corners = []
            for corner in corners:
                # Add homogeneous coordinate
                homogeneous = np.array([corner[0], corner[1], 1.0])
                # Apply inverse homography
                original_homogeneous = inv_homography.dot(homogeneous)
                # Convert back from homogeneous coordinates
                original_homogeneous /= original_homogeneous[2]
                original_corners.append((int(original_homogeneous[0]), int(original_homogeneous[1])))
            
            # Convert to bounding box
            original_corners = np.array(original_corners)
            x_min = np.min(original_corners[:, 0])
            y_min = np.min(original_corners[:, 1])
            x_max = np.max(original_corners[:, 0])
            y_max = np.max(original_corners[:, 1])
            
            original_bottlenecks.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return original_bottlenecks
    
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
            
    def draw_bottleneck(self, frame, is_original_view=False, original_bottlenecks=None):
        """
        Draw bottleneck indications on frame
        
        Parameters:
        - frame: The frame to draw on (modified in-place)
        - is_original_view: Whether this is the original view rather than top view
        - original_bottlenecks: Bottleneck coords in original view (only needed if is_original_view=True)
        """
        if not self.bottlenecks:
            return
            
        # Determine which bottleneck coordinates to use
        bottlenecks_to_draw = original_bottlenecks if is_original_view and original_bottlenecks else self.bottlenecks
        
        # Skip if no valid bottlenecks to draw
        if not bottlenecks_to_draw:
            return
            
        # Create pulsating effect for the bottlenecks
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Draw each bottleneck with a unique color/phase
        for i, bottleneck in enumerate(bottlenecks_to_draw):
            x, y, width, height = bottleneck
            
            # Different pulse phase for each bottleneck
            phase_offset = i * (2 * np.pi / 3)  # 120-degree phase difference
            pulse_value = int(127 + 127 * np.sin(current_time * 2 + phase_offset))
            
            # Color varies for each bottleneck
            # First bottleneck: magenta, second: cyan, third: yellow
            if i % 3 == 0:
                color = (pulse_value, 0, pulse_value)  # Magenta in BGR
            elif i % 3 == 1:
                color = (pulse_value, pulse_value, 0)  # Cyan in BGR
            else:
                color = (0, pulse_value, pulse_value)  # Yellow in BGR
            
            # Draw dashed rectangle around bottleneck area
            line_thickness = 3
            dash_length = 10
            
            # Calculate corner points
            x2, y2 = x + width, y + height
            
            # Draw top horizontal line (dashed)
            for j in range(x, x2, dash_length * 2):
                end = min(j + dash_length, x2)
                cv2.line(frame, (j, y), (end, y), color, line_thickness)
                
            # Draw bottom horizontal line (dashed)
            for j in range(x, x2, dash_length * 2):
                end = min(j + dash_length, x2)
                cv2.line(frame, (j, y2), (end, y2), color, line_thickness)
                
            # Draw left vertical line (dashed)
            for j in range(y, y2, dash_length * 2):
                end = min(j + dash_length, y2)
                cv2.line(frame, (x, j), (x, end), color, line_thickness)
                
            # Draw right vertical line (dashed)
            for j in range(y, y2, dash_length * 2):
                end = min(j + dash_length, y2)
                cv2.line(frame, (x2, j), (x2, end), color, line_thickness)
                
            # Add "BOTTLENECK" text with background for visibility
            text = f"BOTTLENECK {i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Position text at the top of the bottleneck
            text_x = x + (width - text_size[0]) // 2
            text_y = y - 10
            
            # If text would be off-screen, move it inside the box
            if text_y < text_size[1] + 5:
                text_y = y + text_size[1] + 5
                
            # Draw text background for better visibility
            cv2.rectangle(frame, 
                         (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), 
                         (0, 0, 0), -1)
                         
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), 
                       font, font_scale, color, font_thickness)
                       
            # Add anomaly count inside the box
            count_text = f"Anomalies: {len(self.anomalies)}"
            count_size = cv2.getTextSize(count_text, font, font_scale * 0.8, font_thickness)[0]
            
            # Position count text at the bottom of the bottleneck
            count_x = x + (width - count_size[0]) // 2
            count_y = y + height + count_size[1] + 5
            
            # If count text would be off-screen, move it inside the box
            if count_y > frame.shape[0] - 5:
                count_y = y + height - 10
                
            # Draw count background
            cv2.rectangle(frame, 
                         (count_x - 5, count_y - count_size[1] - 5), 
                         (count_x + count_size[0] + 5, count_y + 5), 
                         (0, 0, 0), -1)
                         
            # Draw count text
            cv2.putText(frame, count_text, (count_x, count_y), 
                       font, font_scale * 0.8, color, font_thickness)
        
        # Add stampede warning if active
        if self.stampede_warning:
            warning_text = "STAMPEDE WARNING!"
            warning_size = cv2.getTextSize(warning_text, font, font_scale * 1.2, font_thickness+1)[0]
            
            # Position warning text at the top of the frame (centered)
            warning_x = (frame.shape[1] - warning_size[0]) // 2
            warning_y = 50
            
            # Create flashing effect
            flash_rate = 4  # Faster flashing for urgency
            flash_value = int(200 + 55 * np.sin(current_time * flash_rate))
            
            # Draw warning background
            cv2.rectangle(frame, 
                         (warning_x - 10, warning_y - warning_size[1] - 10), 
                         (warning_x + warning_size[0] + 10, warning_y + 10), 
                         (0, 0, flash_value), -1)  # Blue background
            
            # Draw warning border
            cv2.rectangle(frame, 
                         (warning_x - 10, warning_y - warning_size[1] - 10), 
                         (warning_x + warning_size[0] + 10, warning_y + 10), 
                         (0, 0, 0), 2)  # Black border
            
            # Draw warning text
            cv2.putText(frame, warning_text, (warning_x, warning_y), 
                       font, font_scale * 1.2, (255, 255, 255), font_thickness+1) 