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
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize YOLO model
        print(f"Initializing YOLOv8 on {self.device}...")
        try:
            # Load custom trained model or fallback to standard model based on size
            model_path = "best_re_final.pt"
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            print("Please check your installation and ensure the model is available.")
            raise
        
        # Person class ID (0 for COCO dataset)
        self.person_class_id = 0
        
        # Tracking setup
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
        """
        Enable or disable object tracking
        
        Parameters:
        - enable: Boolean to enable or disable tracking
        """
        self.tracker_enabled = enable
        print(f"Object tracking {'enabled' if enable else 'disabled'}")
        
        if not enable:
            print("Tracking visualization disabled but IDs will still be maintained")
    
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
            # Run YOLOv8 with tracking to get IDs
            results = self.model.track(frame, persist=True, classes=[self.person_class_id], verbose=False)
            
            if not results or len(results) == 0:
                return detections, annotated_frame
                
            # Get the first result (only one image processed)
            result = results[0]
            
            # Check if boxes exist
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                return detections, annotated_frame
            
            # Get IDs if available
            track_ids = None
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            
            # Extract boxes, confidences, and classes
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # Update timeout counters for all existing IDs
            self._update_id_timeouts()
            
            # Process each detection
            for i, box in enumerate(boxes):
                # Only process person class (class_id = 0)
                class_id = int(classes[i])
                if class_id != self.person_class_id:
                    continue
                
                # Get detection details
                x1, y1, x2, y2 = box
                confidence = confidences[i]
                
                # Skip low confidence detections
                if confidence < 0.1:
                    continue
                
                # Get or assign tracking ID
                track_id = self._get_track_id(track_ids, i, x1, y1, x2, y2)
                
                # Add detection to list with ID
                detections.append([
                    int(x1), int(y1), int(x2), int(y2), 
                    float(confidence), 
                    track_id
                ])
                
                # Update tracking history
                self._update_track_history(track_id, x1, y1, x2, y2)
                
                # Visualize the detection on the frame
                self._draw_detection(annotated_frame, x1, y1, x2, y2, confidence, track_id)
            
            # Draw tracking lines if enabled
            if self.tracker_enabled:
                self._draw_tracking_lines(annotated_frame)
                
        except Exception as e:
            print(f"Error in detection: {e}")
            import traceback
            traceback.print_exc()
        
        return detections, annotated_frame
    
    def _update_id_timeouts(self):
        """Update timeout counters for all existing IDs"""
        for id_key in list(self.id_counters.keys()):
            self.id_counters[id_key] += 1
            # Remove IDs that haven't been seen for a while
            if self.id_counters[id_key] > self.id_timeout:
                del self.id_counters[id_key]
                if id_key in self.id_positions:
                    del self.id_positions[id_key]
    
    def _get_track_id(self, track_ids, idx, x1, y1, x2, y2):
        """Get or assign a tracking ID for a detection"""
        # If tracking ID is available from the model, use it
        if track_ids is not None:
            track_id = int(track_ids[idx])
            if track_id is not None:
                # Update position and reset timeout
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                self.id_positions[track_id] = (center_x, center_y)
                self.id_counters[track_id] = 0
                return track_id
        
        # Otherwise, assign our own ID based on spatial proximity
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Find the closest existing ID
        closest_id = None
        min_distance = float('inf')
        
        for id_key, pos in self.id_positions.items():
            pos_x, pos_y = pos
            distance = ((center_x - pos_x) ** 2 + (center_y - pos_y) ** 2) ** 0.5
            # Only consider positions that are close (within 50 pixels)
            if distance < 50 and distance < min_distance:
                closest_id = id_key
                min_distance = distance
        
        if closest_id is not None:
            # Use the existing ID
            track_id = closest_id
            # Reset the timeout counter
            self.id_counters[track_id] = 0
        else:
            # Assign a new ID
            track_id = self.next_id
            self.next_id += 1
            # Initialize the timeout counter
            self.id_counters[track_id] = 0
        
        # Update the position for this ID
        self.id_positions[track_id] = (center_x, center_y)
        
        return track_id
    
    def _update_track_history(self, track_id, x1, y1, x2, y2):
        """Update the tracking history for an ID"""
        if track_id is not None:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Add to track history
            self.track_history[track_id].append((int(center_x), int(center_y)))
            
            # Keep only the last N points
            if len(self.track_history[track_id]) > self.max_track_history:
                self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
    
    def _draw_detection(self, frame, x1, y1, x2, y2, confidence, track_id):
        """Draw detection bounding box and ID on frame"""
        # Get color based on ID
        color = self._get_color_by_id(track_id) if track_id is not None else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    color, 2)
        
        # Add ID text
        id_text = f"ID:{track_id}" if track_id is not None else "No ID"
        label = f"Person {id_text}: {confidence:.2f}"
        cv2.putText(frame, 
                label, 
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 2)
    
    def _draw_tracking_lines(self, frame):
        """Draw tracking lines for visible people"""
        # Only draw tracking for currently visible IDs
        for track_id, track in self.track_history.items():
            if len(track) > 1:
                color = self._get_color_by_id(track_id)
                for i in range(1, len(track)):
                    cv2.line(frame, 
                            track[i-1], 
                            track[i], 
                            color, 2)
    
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

import cv2
import numpy as np
import os
import json
import hashlib
from pathlib import Path

class AreaManager:
    """
    Enhanced class for managing and visualizing areas of interest
    with functionality to save and load defined areas, perspective points,
    and tracking data for detected objects and density points.
    """
    def __init__(self, video_path=None, save_dir="video_data"):
        self.walking_areas = []  # List of polygons defining walking areas
        self.roads = []          # List of polygons defining roads
        self.perspective_points = []  # Four points used for perspective transformation
        self.detected_objects = {}  # Dictionary to store detected objects by frame
        self.density_points = {}  # Dictionary to store density points by frame
        
        # Save the video path for later reference
        self.video_path = video_path
        self.video_id = self._get_video_id(video_path) if video_path else None
        
        # Create the save directory structure if it doesn't exist
        self.save_dir = save_dir
        self.areas_dir = os.path.join(save_dir, "areas")
        self.objects_dir = os.path.join(save_dir, "objects")
        self.density_dir = os.path.join(save_dir, "density")
        self.perspective_dir = os.path.join(save_dir, "perspective")
        
        for directory in [self.save_dir, self.areas_dir, self.objects_dir, self.density_dir, self.perspective_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def _get_video_id(self, video_path):
        """Generate a unique identifier for a video based on its path and file stats"""
        if not video_path:
            return None
            
        # Get file stats (size, modification time)
        try:
            stats = os.stat(video_path)
            # Create a unique identifier based on path, size and modification time
            video_id = f"{os.path.basename(video_path)}_{stats.st_size}_{int(stats.st_mtime)}"
            # Hash it to get a fixed-length string
            return hashlib.md5(video_id.encode()).hexdigest()
        except Exception as e:
            print(f"Warning: Could not get video stats: {e}")
            # Fallback to just the filename
            return hashlib.md5(os.path.basename(video_path).encode()).hexdigest()
    
    def get_frame_hash(self, frame):
        """Generate a unique identifier for a frame based on its content"""
        # Resize the frame to a smaller size to make hashing faster
        small_frame = cv2.resize(frame, (64, 36))
        # Calculate a hash of the frame
        return hashlib.md5(small_frame.tobytes()).hexdigest()
    
    def _get_areas_file_path(self):
        """Get the file path for saving/loading areas based on video ID"""
        if not self.video_id:
            raise ValueError("Video ID not set. Initialize with a valid video path.")
        return os.path.join(self.areas_dir, f"areas_{self.video_id}.json")
    
    def _get_perspective_file_path(self):
        """Get the file path for saving/loading perspective points based on video ID"""
        if not self.video_id:
            raise ValueError("Video ID not set. Initialize with a valid video path.")
        return os.path.join(self.perspective_dir, f"perspective_{self.video_id}.json")
    
    def _get_objects_file_path(self, frame_number=None):
        """Get the file path for saving/loading detected objects based on video ID and frame number"""
        if not self.video_id:
            raise ValueError("Video ID not set. Initialize with a valid video path.")
        
        # If frame number is provided, create a file for that specific frame
        if frame_number is not None:
            return os.path.join(self.objects_dir, f"objects_{self.video_id}_frame_{frame_number:06d}.json")
        
        # Otherwise return the base path for the video
        return os.path.join(self.objects_dir, f"objects_{self.video_id}.json")
    
    def _get_density_file_path(self, frame_number=None):
        """Get the file path for saving/loading density points based on video ID and frame number"""
        if not self.video_id:
            raise ValueError("Video ID not set. Initialize with a valid video path.")
        
        # If frame number is provided, create a file for that specific frame
        if frame_number is not None:
            return os.path.join(self.density_dir, f"density_{self.video_id}_frame_{frame_number:06d}.json")
        
        # Otherwise return the base path for the video
        return os.path.join(self.density_dir, f"density_{self.video_id}.json")
    
    def save_areas(self):
        """Save defined walking areas and roads to a JSON file"""
        if not self.video_id:
            print("Error: Video ID not set. Unable to save areas.")
            return False
        
        # Check if we have any areas to save
        if not self.walking_areas and not self.roads:
            print("Warning: No areas defined to save")
            # Still create an empty file to indicate this video was processed
        
        # Convert numpy arrays to lists for JSON serialization
        areas_data = {
            "walking_areas": [area.tolist() for area in self.walking_areas],
            "roads": [road.tolist() for road in self.roads],
            "video_path": self.video_path
        }
        
        file_path = self._get_areas_file_path()
        print(f"Saving areas to: {os.path.abspath(file_path)}")
        
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
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
    
    def load_areas(self):
        """Load defined walking areas and roads from a JSON file"""
        if not self.video_id:
            print("Error: Video ID not set. Unable to load areas.")
            return False
        
        file_path = self._get_areas_file_path()
        if not os.path.exists(file_path):
            print(f"No saved areas found for this video")
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
    
    def save_perspective_points(self, points):
        """Save perspective transformation points to a JSON file"""
        if not self.video_id:
            print("Error: Video ID not set. Unable to save perspective points.")
            return False
        
        self.perspective_points = points
        
        # Convert points to list for JSON serialization
        points_data = {
            "perspective_points": [point.tolist() for point in points] if isinstance(points[0], np.ndarray) else points,
            "video_path": self.video_path
        }
        
        file_path = self._get_perspective_file_path()
        print(f"Saving perspective points to: {os.path.abspath(file_path)}")
        
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(points_data, f)
            
            # Verify the file was created
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"Perspective points saved successfully to {file_path} (size: {file_size} bytes)")
                return True
            else:
                print(f"Error: File was not created at {file_path}")
                return False
        except Exception as e:
            print(f"Error saving perspective points: {e}")
            return False
    
    def load_perspective_points(self):
        """Load perspective transformation points from a JSON file"""
        if not self.video_id:
            print("Error: Video ID not set. Unable to load perspective points.")
            return False
        
        file_path = self._get_perspective_file_path()
        if not os.path.exists(file_path):
            print(f"No saved perspective points found for this video")
            return False
        
        try:
            with open(file_path, 'r') as f:
                points_data = json.load(f)
            
            # Convert lists back to numpy arrays
            self.perspective_points = [np.array(point) for point in points_data.get("perspective_points", [])]
            
            print(f"Loaded perspective points from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading perspective points: {e}")
            return False
    
    def save_detected_objects(self, frame_number, object_data):
        """Save detected objects for a specific frame to a JSON file
        
        Args:
            frame_number: The frame number
            object_data: List of detected objects with their top view coordinates
                         Format: [{"id": id, "orig_bbox": [x1,y1,x2,y2], "top_view": [x,y], "confidence": conf}, ...]
        """
        if not self.video_id:
            print("Error: Video ID not set. Unable to save detected objects.")
            return False
        
        # Store the data internally
        self.detected_objects[frame_number] = object_data
        
        file_path = self._get_objects_file_path(frame_number)
        print(f"Saving detected objects for frame {frame_number} to: {os.path.abspath(file_path)}")
        
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data for JSON serialization
            json_data = {
                "frame_number": frame_number,
                "video_id": self.video_id,
                "video_path": self.video_path,
                "objects": object_data
            }
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f)
            
            return True
        except Exception as e:
            print(f"Error saving detected objects: {e}")
            return False
    
    def load_detected_objects(self, frame_number=None):
        """Load detected objects from a JSON file for a specific frame or all frames
        
        Args:
            frame_number: The specific frame number to load, or None to load all available frames
            
        Returns:
            Dictionary of detected objects by frame number, or list of objects for specific frame
        """
        if not self.video_id:
            print("Error: Video ID not set. Unable to load detected objects.")
            return None
        
        if frame_number is not None:
            # Load a specific frame
            file_path = self._get_objects_file_path(frame_number)
            if not os.path.exists(file_path):
                #print(f"No saved objects found for frame {frame_number}")
                return []
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.detected_objects[frame_number] = data.get("objects", [])
                return data.get("objects", [])
            except Exception as e:
                print(f"Error loading detected objects for frame {frame_number}: {e}")
                return []
        else:
            # Load all available frames
            objects_dir = os.path.dirname(self._get_objects_file_path())
            prefix = f"objects_{self.video_id}_frame_"
            all_objects = {}
            
            try:
                # Find all files matching the pattern
                for file_name in os.listdir(objects_dir):
                    if file_name.startswith(prefix) and file_name.endswith(".json"):
                        # Extract frame number from filename
                        frame_str = file_name[len(prefix):-5]  # Remove prefix and .json
                        try:
                            frame_num = int(frame_str)
                            file_path = os.path.join(objects_dir, file_name)
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            all_objects[frame_num] = data.get("objects", [])
                        except ValueError:
                            continue
                
                self.detected_objects = all_objects
                print(f"Loaded detected objects for {len(all_objects)} frames")
                return all_objects
            except Exception as e:
                print(f"Error loading all detected objects: {e}")
                return {}
    
    def save_density_points(self, frame_number, density_data):
        """Save density points for a specific frame to a JSON file
        
        Args:
            frame_number: The frame number
            density_data: List of density points with their top view coordinates
                          Format: [{"top_view": [x,y], "density_value": value}, ...]
        """
        if not self.video_id:
            print("Error: Video ID not set. Unable to save density points.")
            return False
        
        # Store the data internally
        self.density_points[frame_number] = density_data
        
        file_path = self._get_density_file_path(frame_number)
        #print(f"Saving density points for frame {frame_number} to: {os.path.abspath(file_path)}")
        
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare data for JSON serialization
            json_data = {
                "frame_number": frame_number,
                "video_id": self.video_id,
                "video_path": self.video_path,
                "density_points": density_data
            }
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f)
            
            return True
        except Exception as e:
            print(f"Error saving density points: {e}")
            return False
    
    def load_density_points(self, frame_number=None):
        """Load density points from a JSON file for a specific frame or all frames
        
        Args:
            frame_number: The specific frame number to load, or None to load all available frames
            
        Returns:
            Dictionary of density points by frame number, or list of points for specific frame
        """
        if not self.video_id:
            print("Error: Video ID not set. Unable to load density points.")
            return None
        
        if frame_number is not None:
            # Load a specific frame
            file_path = self._get_density_file_path(frame_number)
            if not os.path.exists(file_path):
                #print(f"No saved density points found for frame {frame_number}")
                return []
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.density_points[frame_number] = data.get("density_points", [])
                return data.get("density_points", [])
            except Exception as e:
                print(f"Error loading density points for frame {frame_number}: {e}")
                return []
        else:
            # Load all available frames
            density_dir = os.path.dirname(self._get_density_file_path())
            prefix = f"density_{self.video_id}_frame_"
            all_points = {}
            
            try:
                # Find all files matching the pattern
                for file_name in os.listdir(density_dir):
                    if file_name.startswith(prefix) and file_name.endswith(".json"):
                        # Extract frame number from filename
                        frame_str = file_name[len(prefix):-5]  # Remove prefix and .json
                        try:
                            frame_num = int(frame_str)
                            file_path = os.path.join(density_dir, file_name)
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            all_points[frame_num] = data.get("density_points", [])
                        except ValueError:
                            continue
                
                self.density_points = all_points
                print(f"Loaded density points for {len(all_points)} frames")
                return all_points
            except Exception as e:
                print(f"Error loading all density points: {e}")
                return {}
    
    def define_areas(self, frame):
        """Check if areas exist for this video, if not define them"""
        print(f"Checking for existing areas for video ID: {self.video_id}")
        
        # Try to load existing areas
        if self.load_areas():
            print("Using previously defined areas")
            return True
        
        # If no areas exist, define them
        print("No previously defined areas found. Let's define them now.")
        self.define_walking_area(frame)
        self.define_road(frame)
        
        # Save the newly defined areas
        print("Now saving the newly defined areas...")
        success = self.save_areas()
        if success:
            print("Areas successfully saved!")
            return True
        else:
            print("WARNING: Failed to save areas. They will need to be defined again next time.")
            return False
    
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