import cv2
import numpy as np
import torch
import colorsys
from collections import defaultdict
import os

# Conditional import for YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics package not found. Installing...")
    import subprocess
    subprocess.call(['pip', 'install', 'ultralytics'])
    from ultralytics import YOLO

class PersonDetector:
    """
    Class for detecting and tracking people using YOLOv8
    """
    def __init__(self, model_size='x', device=None):
        """
        Initialize YOLOv8 for person detection
        
        Parameters:
        - model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x')
        - device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Initializing YOLO11 on {self.device}...")
        try:
            # Use the Ultralytics YOLO model
            self.model = YOLO(f"best_re_final.pt")
        except Exception as e:
            print(f"Error loading YOLO11 model: {e}")
            print("Please check your installation and ensure the model is available.")
            raise
        
        # Set the model to the specified device
        self.model.to(self.device)
        
        # Person class ID (0 for COCO dataset)
        self.person_class_id = 0
        
        # For tracking people - now enabled by default
        self.tracker_enabled = True
        self.tracked_people = {}
        self.track_history = defaultdict(lambda: [])
        self.max_track_history = 30  # Maximum number of points to keep in track history
        
        # For ID assignment without formal tracking
        self.next_id = 1
        self.detection_ids = {}  # Store IDs for detections based on position
        self.id_positions = {}   # Store last known positions for IDs
        self.id_timeout = 30     # Frames before an unused ID is recycled
        self.id_counters = {}    # Count frames since ID was last seen
    
    def enable_tracking(self, enable=True):
        """Enable or disable object tracking"""
        self.tracker_enabled = enable
        print(f"Object tracking {'enabled' if enable else 'disabled'}")
        
        # If disabling tracking, we don't clear history anymore
        # This allows us to keep IDs even when tracking is disabled
        if not enable:
            print("Tracking disabled but IDs will still be maintained")
    
    def detect(self, frame):
        """
        Detect people in the frame using YOLOv8
        
        Parameters:
        - frame: Input frame (BGR format)
        
        Returns:
        - detections: List of person detections [x1, y1, x2, y2, confidence, id]
        - annotated_frame: Frame with bounding boxes and tracking visualization
        """
        # Make a copy of the frame for annotation
        annotated_frame = frame.copy()
        detections = []
        
        try:
            # Always run YOLOv8 with tracking to get IDs, regardless of tracker_enabled setting
            results = self.model.track(frame, persist=True, classes=[self.person_class_id], verbose=False)
            
            if len(results) > 0:
                # Get the first result (only one image processed)
                result = results[0]
                
                # Check if boxes exist
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    return detections, annotated_frame
                
                # Get IDs if available
                track_ids = None
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()
                
                # Extract boxes
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Update timeout counters for all existing IDs
                for id_key in list(self.id_counters.keys()):
                    self.id_counters[id_key] += 1
                    # Remove IDs that haven't been seen for a while
                    if self.id_counters[id_key] > self.id_timeout:
                        del self.id_counters[id_key]
                        if id_key in self.id_positions:
                            del self.id_positions[id_key]
                
                # Process each detected person
                for i, box in enumerate(boxes):
                    class_id = int(classes[i])
                    
                    # Only process person class
                    if class_id != self.person_class_id:
                        continue
                    
                    # Get detection details
                    x1, y1, x2, y2 = box
                    confidence = confidences[i]
                    
                    # Skip low confidence detections
                    if confidence < 0.1:
                        continue
                    
                    # Get ID if available from tracker
                    track_id = None
                    if track_ids is not None:
                        track_id = int(track_ids[i])
                    
                    # If no track ID is assigned yet by the tracker, assign our own temporary ID
                    # or use previously assigned ID based on spatial proximity
                    if track_id is None:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Find the closest existing ID
                        closest_id = None
                        min_distance = float('inf')
                        
                        for id_key, pos in self.id_positions.items():
                            pos_x, pos_y = pos
                            distance = ((center_x - pos_x) ** 2 + (center_y - pos_y) ** 2) ** 0.5
                            # Only consider positions that are reasonably close (e.g., within 50 pixels)
                            if distance < 50 and distance < min_distance:
                                closest_id = id_key
                                min_distance = distance
                        
                        if closest_id is not None:
                            # Use the existing ID
                            track_id = closest_id
                            # Reset the timeout counter for this ID
                            self.id_counters[track_id] = 0
                        else:
                            # Assign a new ID
                            track_id = self.next_id
                            self.next_id += 1
                            # Initialize the timeout counter
                            self.id_counters[track_id] = 0
                        
                        # Update the position for this ID
                        self.id_positions[track_id] = (center_x, center_y)
                    else:
                        # If a track ID is already assigned by the tracker,
                        # update its position and reset its timeout counter
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        self.id_positions[track_id] = (center_x, center_y)
                        self.id_counters[track_id] = 0
                    
                    # Add detection to list with ID
                    detections.append([
                        int(x1), int(y1), int(x2), int(y2), 
                        float(confidence), 
                        track_id
                    ])
                    
                    # Update tracking history for this ID
                    if track_id is not None:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Add to track history
                        self.track_history[track_id].append((int(center_x), int(center_y)))
                        
                        # Keep only the last N points
                        if len(self.track_history[track_id]) > self.max_track_history:
                            self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
                    
                    # Get color based on ID
                    color = self._get_color_by_id(track_id) if track_id is not None else (0, 255, 0)
                    
                    # Draw detection on frame
                    cv2.rectangle(annotated_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                color, 2)
                    
                    # Add ID text
                    id_text = f"ID:{track_id}" if track_id is not None else "No ID"
                    label = f"Person {id_text}: {confidence:.2f}"
                    cv2.putText(annotated_frame, 
                            label, 
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
                
                # Draw tracking lines if tracking visualization is enabled
                if self.tracker_enabled:
                    # Only draw tracking for currently visible IDs
                    current_ids = [det[5] for det in detections if det[5] is not None]
                    
                    for track_id in current_ids:
                        track = self.track_history.get(track_id, [])
                        if len(track) > 1:
                            color = self._get_color_by_id(track_id)
                            for i in range(1, len(track)):
                                cv2.line(annotated_frame, 
                                        track[i-1], 
                                        track[i], 
                                        color, 2)
            
        except Exception as e:
            print(f"Error in detection: {e}")
            import traceback
            traceback.print_exc()
            # Continue with empty detections
        
        return detections, annotated_frame
    
    def _get_color_by_id(self, track_id):
        """Generate a unique color based on track ID"""
        if track_id is None:
            return (0, 255, 0)  # Default green for no ID
        
        hue = (track_id * 0.1) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

import hashlib
import json

import cv2
import numpy as np
import colorsys
import os
import json
import hashlib

class AreaManager:
    """
    Class for managing and visualizing areas of interest
    with functionality to save and load defined areas
    """
    def __init__(self, save_dir="bounding_box_data"):
        self.walking_areas = []  # List of polygons defining walking areas
        self.roads = []          # List of polygons defining roads
        self.save_dir = save_dir
        
        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory: {self.save_dir}")
    
    def get_frame_hash(self, frame):
        """Generate a unique identifier for a frame based on its content"""
        # Resize the frame to a smaller size to make hashing faster
        small_frame = cv2.resize(frame, (64, 36))
        # Calculate a hash of the frame
        return hashlib.md5(small_frame.tobytes()).hexdigest()
    
    def get_file_path(self, frame_hash):
        """Get the file path for saving/loading areas based on frame hash"""
        return os.path.join(self.save_dir, f"areas_{frame_hash}.json")
    
    def save_areas(self, frame_hash=None, frame=None):
        """Save defined areas to a JSON file"""
        # If frame is provided but not hash, generate hash from frame
        if frame_hash is None and frame is not None:
            frame_hash = self.get_frame_hash(frame)
        
        if frame_hash is None:
            print("Error: Either frame_hash or frame must be provided to save areas")
            return False
        
        # Check if we have any areas to save
        if not self.walking_areas and not self.roads:
            print("Warning: No areas defined to save")
            # Still create an empty file to indicate this frame was processed
        
        # Convert numpy arrays to lists for JSON serialization
        areas_data = {
            "walking_areas": [area.tolist() for area in self.walking_areas],
            "roads": [road.tolist() for road in self.roads]
        }
        
        file_path = self.get_file_path(frame_hash)
        print(f"Attempting to save areas to: {os.path.abspath(file_path)}")
        
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Debug: print what we're about to save
            print(f"Saving {len(self.walking_areas)} walking areas and {len(self.roads)} roads")
            
            with open(file_path, 'w') as f:
                json.dump(areas_data, f)
            
            # Verify the file was created
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"Areas saved successfully to {file_path} (size: {file_size} bytes)")
                return True
            else:
                print(f"Error: File was not created at {file_path}")
                return False
        except Exception as e:
            import traceback
            print(f"Error saving areas: {e}")
            traceback.print_exc()
            return False
    
    def load_areas(self, frame_hash=None, frame=None):
        """Load defined areas from a JSON file"""
        # If frame is provided but not hash, generate hash from frame
        if frame_hash is None and frame is not None:
            frame_hash = self.get_frame_hash(frame)
        
        if frame_hash is None:
            print("Error: Either frame_hash or frame must be provided to load areas")
            return False
        
        file_path = self.get_file_path(frame_hash)
        if not os.path.exists(file_path):
            print(f"No saved areas found for this frame")
            return False
        
        try:
            with open(file_path, 'r') as f:
                areas_data = json.load(f)
            
            # Convert lists back to numpy arrays
            self.walking_areas = [np.array(area) for area in areas_data.get("walking_areas", [])]
            self.roads = [np.array(road) for road in areas_data.get("roads", [])]
            
            print(f"Loaded {len(self.walking_areas)} walking areas and {len(self.roads)} roads from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading areas: {e}")
            return False
    
    def define_areas(self, frame):
        """Check if areas exist for this frame, if not define them"""
        frame_hash = self.get_frame_hash(frame)
        print(f"Frame hash: {frame_hash}")
        
        # Try to load existing areas
        if self.load_areas(frame_hash=frame_hash):
            print("Using previously defined areas")
            return
        
        # If no areas exist, define them
        print("No previously defined areas found. Let's define them now.")
        self.define_walking_area(frame)
        self.define_road(frame)
        
        # Save the newly defined areas
        print("Now saving the newly defined areas...")
        success = self.save_areas(frame_hash=frame_hash)
        if success:
            print("Areas successfully saved!")
        else:
            print("WARNING: Failed to save areas. They will need to be defined again next time.")
    
    def define_walking_area(self, frame):
        """Let user define walking area boundaries on the original frame"""
        frame_copy = frame.copy()
        
        area_points = []
        temp_frame = frame_copy.copy()
        
        def click_event(event, x, y, flags, params):
            nonlocal temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                area_points.append((x, y))
                # Draw point
                cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
                # Draw lines connecting points
                if len(area_points) > 1:
                    cv2.line(temp_frame, area_points[-2], area_points[-1], (0, 255, 0), 2)
                cv2.imshow('Define Walking Area (Right-click to complete)', temp_frame)
            
            elif event == cv2.EVENT_RBUTTONDOWN and len(area_points) > 2:
                # Complete the polygon by connecting back to the first point
                cv2.line(temp_frame, area_points[-1], area_points[0], (0, 255, 0), 2)
                # Show the completed polygon
                overlay = temp_frame.copy()
                cv2.fillPoly(overlay, [np.array(area_points)], (0, 255, 0, 128))
                cv2.addWeighted(overlay, 0.3, temp_frame, 0.7, 0, temp_frame)
                cv2.imshow('Define Walking Area (Right-click to complete)', temp_frame)
                # Add walking area to the list
                self.walking_areas.append(np.array(area_points))
                print(f"Walking area defined with {len(area_points)} points")
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
        
        print("Define walking area: Left-click to add points, Right-click to complete")
        cv2.imshow('Define Walking Area (Right-click to complete)', frame_copy)
        cv2.setMouseCallback('Define Walking Area (Right-click to complete)', click_event)
        
        # Wait until walking area is defined
        while not self.walking_areas:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def define_road(self, frame):
        """Let user define road areas on the original frame"""
        # Create a canvas showing walking areas
        frame_copy = frame.copy()
        
        # Draw existing walking areas
        for area in self.walking_areas:
            overlay = frame_copy.copy()
            cv2.fillPoly(overlay, [area], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)
            cv2.polylines(frame_copy, [area], True, (0, 255, 0), 2)
        
        road_points = []
        temp_frame = frame_copy.copy()
        
        def click_event(event, x, y, flags, params):
            nonlocal temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                road_points.append((x, y))
                # Draw point
                cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
                # Draw lines connecting points
                if len(road_points) > 1:
                    cv2.line(temp_frame, road_points[-2], road_points[-1], (0, 0, 255), 2)
                cv2.imshow('Define Road (Right-click to complete, press Q when done)', temp_frame)
            
            elif event == cv2.EVENT_RBUTTONDOWN and len(road_points) > 2:
                # Complete the polygon by connecting back to the first point
                cv2.line(temp_frame, road_points[-1], road_points[0], (0, 0, 255), 2)
                # Show the completed polygon
                overlay = temp_frame.copy()
                cv2.fillPoly(overlay, [np.array(road_points)], (0, 0, 255, 128))
                cv2.addWeighted(overlay, 0.3, temp_frame, 0.7, 0, temp_frame)
                cv2.imshow('Define Road (Right-click to complete, press Q when done)', temp_frame)
                # Add road to the list
                self.roads.append(np.array(road_points))
                print(f"Road defined with {len(road_points)} points")
                # Clear points for next road
                road_points.clear()
                temp_frame = frame_copy.copy()
                # Draw existing roads
                for road in self.roads:
                    overlay = temp_frame.copy()
                    cv2.fillPoly(overlay, [road], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.3, temp_frame, 0.7, 0, temp_frame)
                    cv2.polylines(temp_frame, [road], True, (0, 0, 255), 2)
                cv2.imshow('Define Road (Right-click to complete, press Q when done)', temp_frame)
        
        print("Define roads: Left-click to add points, Right-click to complete, press Q when done")
        cv2.imshow('Define Road (Right-click to complete, press Q when done)', frame_copy)
        cv2.setMouseCallback('Define Road (Right-click to complete, press Q when done)', click_event)
        
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass
        
        cv2.destroyAllWindows()
    
    def draw_on_frame(self, frame):
        """Draw walking areas and roads on the frame"""
        overlay = frame.copy()
        
        # Draw roads (red with transparency)
        for road in self.roads:
            cv2.fillPoly(overlay, [road], (0, 0, 255))
            cv2.polylines(frame, [road], True, (0, 0, 255), 2)
        
        # Draw walking areas (green with transparency)
        for area in self.walking_areas:
            cv2.fillPoly(overlay, [area], (0, 255, 0))
            cv2.polylines(frame, [area], True, (0, 255, 0), 2)
        
        # Apply the overlay with transparency
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add legend
        cv2.rectangle(frame, (10, 10), (200, 70), (255, 255, 255), -1)
        cv2.rectangle(frame, (10, 10), (200, 70), (0, 0, 0), 1)
        cv2.putText(frame, "Legend:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (15, 40), (30, 55), (0, 255, 0), -1)
        cv2.putText(frame, "Walking Area", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (120, 40), (135, 55), (0, 0, 255), -1)
        cv2.putText(frame, "Road", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame