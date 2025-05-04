import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import time
import subprocess
import sys
from pathlib import Path
import json
import hashlib

# Import from existing codebase
from visualization import process_cctv_to_top_view, get_perspective_transform
from simulation_enhancement import enhanced_process_cctv_to_top_view
from person_detection import AreaManager, PersonDetector
from density_estimation import CrowdDensityEstimator
from anomaly_detection import AnomalyDetector
from video_preprocessor import preprocess_video

# Set page config
st.set_page_config(
    page_title="BhedChaal Simple Interactive - CCTV Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define global variables
TEMP_DIR = "temp"
BBOX_DATA_DIR = "bounding_box_data"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Initialize session state variables
if 'points' not in st.session_state:
    st.session_state.points = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'perspective_points' not in st.session_state:
    st.session_state.perspective_points = []
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_point_index' not in st.session_state:
    st.session_state.current_point_index = 0
if 'points_description' not in st.session_state:
    st.session_state.points_description = [
        "Top Left",
        "Top Right",
        "Bottom Right",
        "Bottom Left"
    ]
if 'walking_areas' not in st.session_state:
    st.session_state.walking_areas = []
if 'roads' not in st.session_state:
    st.session_state.roads = []
if 'current_area_points' not in st.session_state:
    st.session_state.current_area_points = []
if 'area_type' not in st.session_state:
    st.session_state.area_type = 'walking'
if 'coordinates_saved' not in st.session_state:
    st.session_state.coordinates_saved = False

def get_video_id(video_path):
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

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the file path"""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def check_saved_coordinates(video_path):
    """Check if coordinates are already saved for this video in bounding_box_data directory"""
    video_id = get_video_id(video_path)
    areas_file = os.path.join(BBOX_DATA_DIR, f"areas_{video_id}.json")
    
    # First check in bounding_box_data directory
    if os.path.exists(areas_file):
        try:
            with open(areas_file, 'r') as f:
                areas_data = json.load(f)
            
            # Load walking areas and roads
            st.session_state.walking_areas = areas_data.get("walking_areas", [])
            st.session_state.roads = areas_data.get("roads", [])
            
            # Check if walking areas exist and perspective points exist in areas_data
            if st.session_state.walking_areas:
                # Check if perspective_points are saved in the JSON
                if "perspective_points" in areas_data and len(areas_data["perspective_points"]) == 4:
                    st.session_state.perspective_points = areas_data["perspective_points"]
                # Else use the first 4 points of the first walking area as perspective points if needed
                elif len(st.session_state.walking_areas[0]) >= 4:
                    st.session_state.perspective_points = st.session_state.walking_areas[0][:4]
                
                # Always set the points to the perspective points for display
                if st.session_state.perspective_points:
                    st.session_state.points = st.session_state.perspective_points.copy()
                    print(f"Loaded coordinates from {areas_file}")
                    st.session_state.coordinates_saved = True
                    return True
                    
        except Exception as e:
            print(f"Error loading saved coordinates: {e}")
    
    # If not found in bounding_box_data, check in video_data
    area_manager = AreaManager(video_path=video_path)
    if area_manager.load_areas() and area_manager.load_perspective_points():
        st.session_state.walking_areas = [area.tolist() for area in area_manager.walking_areas]
        st.session_state.roads = [road.tolist() for road in area_manager.roads]
        st.session_state.perspective_points = [point.tolist() for point in area_manager.perspective_points]
        st.session_state.points = st.session_state.perspective_points.copy()
        
        # Also save to bounding_box_data to ensure it's available there
        save_coordinates_to_bounding_box_data(video_path)
        
        st.session_state.coordinates_saved = True
        return True
    
    return False

def save_coordinates_to_bounding_box_data(video_path):
    """Save current coordinates to bounding_box_data directory"""
    video_id = get_video_id(video_path)
    
    # Ensure the directory exists
    os.makedirs(BBOX_DATA_DIR, exist_ok=True)
    
    # Save to bounding_box_data directory
    areas_file = os.path.join(BBOX_DATA_DIR, f"areas_{video_id}.json")
    
    # Create data structure
    areas_data = {
        "walking_areas": st.session_state.walking_areas,
        "roads": st.session_state.roads,
        "perspective_points": st.session_state.perspective_points
    }
    
    # Save to file
    try:
        with open(areas_file, 'w') as f:
            json.dump(areas_data, f)
        
        print(f"Saved coordinates to {areas_file}")
        st.session_state.coordinates_saved = True
        return True
    except Exception as e:
        print(f"Error saving coordinates: {e}")
        return False

def display_visual_point_selection(frame):
    """
    Display an interactive way to select points by showing preview and providing UI
    """
    st.write("### Step 1: Select 4 points for perspective transformation")
    st.write("Select points in the following order: top-left, top-right, bottom-right, bottom-left")
    
    # Get image dimensions
    h, w = frame.shape[:2]
    
    # Calculate display width maintaining aspect ratio
    display_width = 800
    display_height = int(h * display_width / w)
    
    # Create two columns - one for the image, one for the controls
    col1, col2 = st.columns([3, 1])
    
    # Copy the frame for drawing
    preview_img = frame.copy()
    
    # Draw existing points on the preview image
    for i, point in enumerate(st.session_state.points):
        # Draw points on image
        cv2.circle(preview_img, tuple(point), 8, (0, 255, 0), -1)
        cv2.putText(preview_img, f"Point {i+1}", (point[0]+10, point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw lines connecting points
        if i > 0:
            cv2.line(preview_img, tuple(st.session_state.points[i-1]), tuple(point), (0, 255, 0), 2)
        
        # Connect last point to first if we have 4 points
        if i == 3:
            cv2.line(preview_img, tuple(point), tuple(st.session_state.points[0]), (0, 255, 0), 2)
    
    with col1:
        # Display the image
        st.image(preview_img, caption="Click a point location and then press 'Add Point'", 
                width=display_width, channels="BGR", use_column_width=False)
    
    with col2:
        st.write("### Point Selection")
        
        # Show which point is being added next
        current_index = min(st.session_state.current_point_index, 3)
        st.write(f"Currently selecting: **{st.session_state.points_description[current_index]}**")
        
        # Create sliders for precise control
        x = st.slider("X coordinate", 0, w, w//2, key="point_x")
        y = st.slider("Y coordinate", 0, h, h//2, key="point_y")
        
        # Button to add the current point
        if st.button("Add Point"):
            if st.session_state.current_point_index < 4:
                # If we're still adding the first 4 points
                if len(st.session_state.points) <= st.session_state.current_point_index:
                    st.session_state.points.append([x, y])
                else:
                    # Replace existing point at current index
                    st.session_state.points[st.session_state.current_point_index] = [x, y]
                
                # Move to next point
                st.session_state.current_point_index += 1
                if st.session_state.current_point_index > 3:
                    st.session_state.current_point_index = 0  # Wrap around for adjustments
                
                st.rerun()
            
        # Button to clear all points
        if st.button("Clear Points"):
            st.session_state.points = []
            st.session_state.current_point_index = 0
            st.rerun()
        
        # Show the currently selected points
        if st.session_state.points:
            st.write("### Selected Points")
            for i, point in enumerate(st.session_state.points):
                st.write(f"{st.session_state.points_description[i]}: ({point[0]}, {point[1]})")
        
        # Button to confirm points
        if st.button("Confirm Points") and len(st.session_state.points) == 4:
            st.session_state.perspective_points = st.session_state.points.copy()
            # Save coordinates after confirmation
            save_coordinates_to_bounding_box_data(st.session_state.video_path)
            st.session_state.step = "areas"
            st.rerun()
    
    return st.session_state.points

def display_area_selection(frame):
    """Display interface for selecting walking areas and roads"""
    st.write("### Step 2: Define Areas")
    st.write("Define walking areas (green) and roads (red)")
    
    # Get image dimensions
    h, w = frame.shape[:2]
    
    # Calculate display width maintaining aspect ratio
    display_width = 800
    display_height = int(h * display_width / w)
    
    # Copy the frame for drawing
    preview_img = frame.copy()
    
    # Draw existing walking areas
    for area in st.session_state.walking_areas:
        area_np = np.array(area, dtype=np.int32)
        cv2.fillPoly(preview_img, [area_np], (0, 255, 0, 128))
        cv2.polylines(preview_img, [area_np], True, (0, 255, 0), 2)
    
    # Draw existing roads
    for road in st.session_state.roads:
        road_np = np.array(road, dtype=np.int32)
        cv2.fillPoly(preview_img, [road_np], (0, 0, 255, 128))
        cv2.polylines(preview_img, [road_np], True, (0, 0, 255), 2)
    
    # Draw current area points
    if st.session_state.current_area_points:
        current_color = (0, 255, 0) if st.session_state.area_type == 'walking' else (0, 0, 255)
        for i, point in enumerate(st.session_state.current_area_points):
            cv2.circle(preview_img, tuple(point), 5, current_color, -1)
            if i > 0:
                cv2.line(preview_img, tuple(st.session_state.current_area_points[i-1]), 
                         tuple(point), current_color, 2)
        
        # Connect last point to first if we have at least 3 points
        if len(st.session_state.current_area_points) >= 3:
            cv2.line(preview_img, tuple(st.session_state.current_area_points[-1]), 
                     tuple(st.session_state.current_area_points[0]), current_color, 2)
    
    # Create two columns - one for the image, one for the controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display the image
        st.image(preview_img, caption="Define areas by adding points and completing them", 
                width=display_width, channels="BGR", use_column_width=False)
    
    with col2:
        st.write("### Area Selection")
        
        # Area type selection
        area_type = st.radio("Area Type", ["Walking Area", "Road"], 
                           index=0 if st.session_state.area_type == 'walking' else 1)
        st.session_state.area_type = 'walking' if area_type == "Walking Area" else 'road'
        
        # Create sliders for precise control
        x = st.slider("X coordinate", 0, w, w//2, key="area_x")
        y = st.slider("Y coordinate", 0, h, h//2, key="area_y")
        
        # Button to add a point to the current area
        if st.button("Add Point to Area"):
            st.session_state.current_area_points.append([x, y])
            st.rerun()
        
        # Button to complete the current area
        if st.button("Complete Area") and len(st.session_state.current_area_points) >= 3:
            if st.session_state.area_type == 'walking':
                st.session_state.walking_areas.append(st.session_state.current_area_points.copy())
            else:
                st.session_state.roads.append(st.session_state.current_area_points.copy())
            
            st.session_state.current_area_points = []
            # Save coordinates whenever an area is completed
            save_coordinates_to_bounding_box_data(st.session_state.video_path)
            st.rerun()
        
        # Button to clear current area points
        if st.button("Clear Current Points"):
            st.session_state.current_area_points = []
            st.rerun()
        
        # Button to clear all areas
        if st.button("Clear All Areas"):
            st.session_state.walking_areas = []
            st.session_state.roads = []
            st.session_state.current_area_points = []
            st.rerun()
        
        # Show statistics
        st.write(f"Walking Areas: {len(st.session_state.walking_areas)}")
        st.write(f"Roads: {len(st.session_state.roads)}")
        st.write(f"Current Points: {len(st.session_state.current_area_points)}")
        
        # Button to continue to options
        if st.button("Continue to Options"):
            # Final save before moving to options
            save_coordinates_to_bounding_box_data(st.session_state.video_path)
            st.session_state.step = "options"
            st.rerun()
    
    # Button to go back to perspective selection
    if st.button("Back to Perspective Selection"):
        st.session_state.step = "perspective"
        st.rerun()

def run_video_analysis(video_path, src_points, options, walking_areas, roads):
    """Run video analysis with the existing code and return results"""
    
    # Create output paths
    video_name = Path(video_path).stem
    output_dir = os.path.join(TEMP_DIR, "output_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    output_original = os.path.join(output_dir, f"{video_name}_enhanced_original.mp4")
    output_top_view = os.path.join(output_dir, f"{video_name}_enhanced_top_view.mp4")
    output_density = os.path.join(output_dir, f"{video_name}_enhanced_density.mp4")
    
    # Create area manager and set areas
    area_manager = AreaManager(video_path=video_path)
    
    # Set walking areas and roads directly if provided (from saved coordinates)
    # This will prevent CLI windows from opening
    if walking_areas and roads:
        area_manager.walking_areas = [np.array(area, np.int32) for area in walking_areas]
        area_manager.roads = [np.array(road, np.int32) for road in roads]
        # Save perspective points as well to prevent CLI windows
        if src_points:
            area_manager.perspective_points = [np.array(point, np.int32) for point in src_points]
            area_manager.save_perspective_points(area_manager.perspective_points)
        area_manager.save_areas()  # Save them to ensure they're available
        # Set the flags to skip CLI area selection windows
        save_data = False  # We already saved the data
        load_saved_data = True  # Load saved data instead of prompting
    else:
        # Try to load from saved data
        area_manager.load_areas()
        # Default flags if we don't have saved coordinates
        save_data = True
        load_saved_data = True
    
    # Process the video
    if options["enhanced"]:
        result = enhanced_process_cctv_to_top_view(
            video_path,
            output_original,
            None,  # Use the first frame as calibration
            src_points,
            use_tracking=options["tracking"],
            yolo_model_size=options["model_size"],
            csrnet_model_path=options.get("csrnet_weights"),
            density_threshold=options["density_threshold"],
            max_points=options["max_points"],
            preprocess_video=options["preprocess"],
            anomaly_threshold=options["anomaly_threshold"],
            stampede_threshold=options["stampede_threshold"],
            max_bottlenecks=options["max_bottlenecks"],
            save_data=save_data,
            load_saved_data=load_saved_data
        )
    else:
        result = process_cctv_to_top_view(
            video_path,
            output_original,
            None,  # Use the first frame as calibration
            src_points,
            use_tracking=options["tracking"],
            yolo_model_size=options["model_size"],
            csrnet_model_path=options.get("csrnet_weights"),
            preprocess_video=options["preprocess"],
            anomaly_threshold=options["anomaly_threshold"],
            stampede_threshold=options["stampede_threshold"],
            max_bottlenecks=options["max_bottlenecks"],
            save_data=save_data,
            load_saved_data=load_saved_data
        )
    
    return {
        "original_video": output_original,
        "top_view_video": output_top_view,
        "density_video": output_density,
        "result": result
    }

def show_results(results):
    """Display the results of the video analysis"""
    
    st.write("## Analysis Results")
    
    # Create three columns for the different videos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### Original Video with Detections")
        if os.path.exists(results["original_video"]):
            st.video(results["original_video"])
        else:
            st.error("Original video output not found.")
    
    with col2:
        st.write("### Top View")
        if os.path.exists(results["top_view_video"]):
            st.video(results["top_view_video"])
        else:
            st.error("Top view video output not found.")
    
    with col3:
        st.write("### Density Visualization")
        if os.path.exists(results["density_video"]):
            st.video(results["density_video"])
        else:
            st.error("Density video output not found.")
    
    # Show additional outputs
    st.write("### Processing Results")
    st.json(results["result"])
    
    # Button to restart
    if st.button("Process Another Video"):
        st.session_state.step = "upload"
        st.session_state.video_path = None
        st.session_state.frame = None
        st.session_state.perspective_points = []
        st.session_state.points = []
        st.session_state.results = None
        st.session_state.current_point_index = 0
        st.session_state.walking_areas = []
        st.session_state.roads = []
        st.session_state.current_area_points = []
        st.rerun()

def main():
    st.title("BhedChaal - Simple Interactive CCTV Crowd Analysis")
    
    # Upload video step
    if st.session_state.step == "upload":
        # Video input
        st.write("### Upload Video")
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = save_uploaded_file(uploaded_file)
            st.session_state.video_path = video_path
            st.write(f"Video uploaded successfully: {uploaded_file.name}")
            
            # Check if we already have saved coordinates
            has_saved_coords = check_saved_coordinates(video_path)
            
            # Get the first frame for perspective selection
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                st.session_state.frame = frame
                # Skip directly to options if coordinates are already set
                if has_saved_coords:
                    st.write("Found previously saved coordinates. Skipping to processing options.")
                    st.session_state.step = "options"
                else:
                    st.session_state.step = "perspective"
                st.rerun()
            else:
                st.error("Failed to read the uploaded video. Please try a different file.")
    
    # Perspective selection step
    elif st.session_state.step == "perspective":
        if st.session_state.frame is not None:
            display_visual_point_selection(st.session_state.frame)
            
            # Button to go back to upload
            if st.button("Back to Upload"):
                st.session_state.step = "upload"
                st.rerun()
    
    # Areas selection step
    elif st.session_state.step == "areas":
        if st.session_state.frame is not None:
            display_area_selection(st.session_state.frame)
    
    # Options selection step
    elif st.session_state.step == "options":
        st.write("### Processing Options")
        
        # Sidebar for options
        st.sidebar.title("Options")
        
        # Processing options
        enhanced = st.sidebar.checkbox("Enhanced Visualization", value=True)
        tracking = st.sidebar.checkbox("Enable Tracking", value=True)
        preprocess = st.sidebar.checkbox("Preprocess Video", value=True)
        
        model_size = st.sidebar.selectbox(
            "YOLOv8 Model Size",
            ["n", "s", "m", "l", "x"],
            index=4  # Default to 'x'
        )
        
        # CSRNet weights selection
        weights_path = st.sidebar.text_input("CSRNet Weights Path", "weights.pth")
        
        # Enhanced options
        density_threshold = 0.2
        max_points = 200
        anomaly_threshold = 30
        stampede_threshold = 35
        max_bottlenecks = 3
        
        if enhanced:
            st.sidebar.header("Enhanced Options")
            density_threshold = st.sidebar.slider(
                "Density Threshold",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05
            )
            
            max_points = st.sidebar.slider(
                "Max Density Points",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )
            
            anomaly_threshold = st.sidebar.slider(
                "Anomaly Threshold",
                min_value=10,
                max_value=50,
                value=30,
                step=5
            )
            
            stampede_threshold = st.sidebar.slider(
                "Stampede Threshold",
                min_value=15,
                max_value=55,
                value=35,
                step=5
            )
            
            max_bottlenecks = st.sidebar.slider(
                "Max Bottlenecks",
                min_value=1,
                max_value=5,
                value=3,
                step=1
            )
        
        # Display selected points preview
        if st.session_state.frame is not None and st.session_state.perspective_points:
            preview_img = st.session_state.frame.copy()
            
            # Draw perspective points
            for i, point in enumerate(st.session_state.perspective_points):
                # Draw points on image
                cv2.circle(preview_img, tuple(point), 5, (0, 255, 0), -1)
                
                # Draw lines connecting points
                if i > 0:
                    cv2.line(preview_img, tuple(st.session_state.perspective_points[i-1]), tuple(point), (0, 255, 0), 2)
                
                # Connect last point to first
                if i == len(st.session_state.perspective_points) - 1:
                    cv2.line(preview_img, tuple(point), tuple(st.session_state.perspective_points[0]), (0, 255, 0), 2)
            
            # Display the preview
            st.image(preview_img, caption="Selected perspective points", width=800, channels="BGR")
        
        # Display areas
        st.write("### Defined Areas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Walking Areas: {len(st.session_state.walking_areas)}")
        
        with col2:
            st.write(f"Roads: {len(st.session_state.roads)}")
        
        # Process button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Area Selection"):
                st.session_state.step = "areas"
                st.rerun()
        
        with col2:
            if st.button("Process Video"):
                with st.spinner("Processing video... This may take a while."):
                    # Collect options
                    options = {
                        "enhanced": enhanced,
                        "tracking": tracking,
                        "model_size": model_size,
                        "preprocess": preprocess,
                        "density_threshold": density_threshold,
                        "max_points": max_points,
                        "anomaly_threshold": anomaly_threshold,
                        "stampede_threshold": stampede_threshold,
                        "max_bottlenecks": max_bottlenecks,
                        "csrnet_weights": weights_path
                    }
                    
                    # Run analysis
                    results = run_video_analysis(
                        st.session_state.video_path, 
                        st.session_state.perspective_points, 
                        options,
                        st.session_state.walking_areas,
                        st.session_state.roads
                    )
                    
                    # Store results and move to results page
                    st.session_state.results = results
                    st.session_state.step = "results"
                    st.rerun()
    
    # Results step
    elif st.session_state.step == "results":
        if st.session_state.results:
            show_results(st.session_state.results)
        else:
            st.error("No results available. Please process a video first.")
            if st.button("Back to Upload"):
                st.session_state.step = "upload"
                st.rerun()

if __name__ == "__main__":
    main() 