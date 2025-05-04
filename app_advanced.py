import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import json
import time
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Import from existing codebase
from visualization import process_cctv_to_top_view, get_perspective_transform
from simulation_enhancement import enhanced_process_cctv_to_top_view
from person_detection import AreaManager, PersonDetector
from density_estimation import CrowdDensityEstimator
from anomaly_detection import AnomalyDetector
from video_preprocessor import preprocess_video

# Set page config
st.set_page_config(
    page_title="BhedChaal Advanced - CCTV Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define global variables
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'upload'  # Default page
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'perspective_points' not in st.session_state:
    st.session_state.perspective_points = None
if 'walking_areas' not in st.session_state:
    st.session_state.walking_areas = []
if 'roads' not in st.session_state:
    st.session_state.roads = []
if 'current_area_points' not in st.session_state:
    st.session_state.current_area_points = []
if 'area_type' not in st.session_state:
    st.session_state.area_type = 'walking'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing_options' not in st.session_state:
    st.session_state.processing_options = {
        "enhanced": True,
        "tracking": True,
        "model_size": "x",
        "preprocess": True,
        "density_threshold": 0.2,
        "max_points": 200,
        "anomaly_threshold": 30,
        "stampede_threshold": 35,
        "max_bottlenecks": 3
    }

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the file path"""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def select_perspective_points(image):
    """Streamlit interface to select 4 points for perspective transformation"""
    st.write("### Select 4 points for perspective transformation")
    st.write("Select in clockwise order: top-left, top-right, bottom-right, bottom-left")
    
    # Display the image for point selection
    h, w = image.shape[:2]
    
    # Calculate display width maintaining aspect ratio
    display_width = 800
    display_height = int(h * display_width / w)
    
    # Create four columns for point coordinates
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("Top Left")
        tl_x = st.number_input("TL X", 0, w, int(w*0.2), key="tl_x")
        tl_y = st.number_input("TL Y", 0, h, int(h*0.2), key="tl_y")
        
    with col2:
        st.write("Top Right")
        tr_x = st.number_input("TR X", 0, w, int(w*0.8), key="tr_x")
        tr_y = st.number_input("TR Y", 0, h, int(h*0.2), key="tr_y")
        
    with col3:
        st.write("Bottom Right")
        br_x = st.number_input("BR X", 0, w, int(w*0.8), key="br_x")
        br_y = st.number_input("BR Y", 0, h, int(h*0.8), key="br_y")
        
    with col4:
        st.write("Bottom Left")
        bl_x = st.number_input("BL X", 0, w, int(w*0.2), key="bl_x")
        bl_y = st.number_input("BL Y", 0, h, int(h*0.8), key="bl_y")
    
    # Display selected points on the image
    preview_img = image.copy()
    points = [(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)]
    
    # Draw points and lines
    for i, (x, y) in enumerate(points):
        cv2.circle(preview_img, (x, y), 5, (0, 255, 0), -1)
        # Connect with next point
        next_point = points[(i + 1) % 4]
        cv2.line(preview_img, (x, y), next_point, (0, 255, 0), 2)
    
    # Update the image
    st.image(preview_img, width=display_width, channels="BGR")
    
    # Return the 4 points
    return [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]

def draw_areas_on_image(image, walking_areas, roads):
    """Draw all defined areas on the image"""
    img_copy = image.copy()
    
    # Draw walking areas in green
    for area in walking_areas:
        area_np = np.array(area, dtype=np.int32)
        cv2.fillPoly(img_copy, [area_np], (0, 255, 0, 128))
        cv2.polylines(img_copy, [area_np], True, (0, 255, 0), 2)
    
    # Draw roads in red
    for road in roads:
        road_np = np.array(road, dtype=np.int32)
        cv2.fillPoly(img_copy, [road_np], (0, 0, 255, 128))
        cv2.polylines(img_copy, [road_np], True, (0, 0, 255), 2)
    
    return img_copy

def select_areas(image):
    """Interface to select walking areas and roads"""
    st.write("### Define Areas")
    st.write("Define walking areas (green) and roads (red) by clicking on the image.")
    
    # Display the image
    h, w = image.shape[:2]
    
    # Calculate display width maintaining aspect ratio
    display_width = 800
    display_height = int(h * display_width / w)
    
    # Area selection interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Area type selection
        area_type = st.radio("Area Type", ["Walking Area", "Road"], 
                             index=0 if st.session_state.area_type == 'walking' else 1,
                             key="area_type_radio")
        st.session_state.area_type = 'walking' if area_type == "Walking Area" else 'road'
        
        # Coordinate inputs
        st.write("### Add Point")
        x = st.number_input("X", 0, w, w//2, key="area_x")
        y = st.number_input("Y", 0, h, h//2, key="area_y")
        
        if st.button("Add Point"):
            st.session_state.current_area_points.append([x, y])
        
        if st.button("Complete Area") and len(st.session_state.current_area_points) >= 3:
            if st.session_state.area_type == 'walking':
                st.session_state.walking_areas.append(st.session_state.current_area_points)
            else:
                st.session_state.roads.append(st.session_state.current_area_points)
            st.session_state.current_area_points = []
            st.rerun()
        
        if st.button("Clear Current Points"):
            st.session_state.current_area_points = []
            st.rerun()
        
        if st.button("Clear All Areas"):
            st.session_state.walking_areas = []
            st.session_state.roads = []
            st.session_state.current_area_points = []
            st.rerun()
    
    with col1:
        # Draw all areas and current selection on the image
        img_with_areas = draw_areas_on_image(image, 
                                           st.session_state.walking_areas, 
                                           st.session_state.roads)
        
        # Draw current points
        if st.session_state.current_area_points:
            current_color = (0, 255, 0) if st.session_state.area_type == 'walking' else (0, 0, 255)
            for i, point in enumerate(st.session_state.current_area_points):
                cv2.circle(img_with_areas, tuple(point), 5, current_color, -1)
                if i > 0:
                    cv2.line(img_with_areas, 
                             tuple(st.session_state.current_area_points[i-1]), 
                             tuple(point), 
                             current_color, 2)
            
            # Connect last point to first if we have at least 3 points
            if len(st.session_state.current_area_points) >= 3:
                cv2.line(img_with_areas, 
                         tuple(st.session_state.current_area_points[-1]), 
                         tuple(st.session_state.current_area_points[0]), 
                         current_color, 2)
        
        # Display the image with areas
        st.image(img_with_areas, width=display_width, channels="BGR")
    
    return st.session_state.walking_areas, st.session_state.roads

def run_video_analysis():
    """Run video analysis with the existing code and return results"""
    # Create area manager and set areas
    area_manager = AreaManager()
    area_manager.walking_areas = [np.array(area, np.int32) for area in st.session_state.walking_areas]
    area_manager.roads = [np.array(road, np.int32) for road in st.session_state.roads]
    
    # Create output paths
    video_name = Path(st.session_state.video_path).stem
    output_dir = os.path.join(TEMP_DIR, "output_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    output_original = os.path.join(output_dir, f"{video_name}_enhanced_original.mp4")
    output_top_view = os.path.join(output_dir, f"{video_name}_enhanced_top_view.mp4")
    output_density = os.path.join(output_dir, f"{video_name}_enhanced_density.mp4")
    
    # Process the video
    options = st.session_state.processing_options
    
    if options["enhanced"]:
        result = enhanced_process_cctv_to_top_view(
            st.session_state.video_path,
            output_original,
            None,  # Use the first frame as calibration
            st.session_state.perspective_points,
            use_tracking=options["tracking"],
            yolo_model_size=options["model_size"],
            csrnet_model_path=options.get("csrnet_weights"),
            density_threshold=options["density_threshold"],
            max_points=options["max_points"],
            preprocess_video=options["preprocess"],
            anomaly_threshold=options["anomaly_threshold"],
            stampede_threshold=options["stampede_threshold"],
            max_bottlenecks=options["max_bottlenecks"],
            area_manager=area_manager
        )
    else:
        result = process_cctv_to_top_view(
            st.session_state.video_path,
            output_original,
            None,  # Use the first frame as calibration
            st.session_state.perspective_points,
            use_tracking=options["tracking"],
            yolo_model_size=options["model_size"],
            csrnet_model_path=options.get("csrnet_weights"),
            preprocess_video=options["preprocess"],
            anomaly_threshold=options["anomaly_threshold"],
            stampede_threshold=options["stampede_threshold"],
            max_bottlenecks=options["max_bottlenecks"],
            area_manager=area_manager
        )
    
    # Results dict
    st.session_state.results = {
        "original_video": output_original,
        "top_view_video": output_top_view,
        "density_video": output_density,
        "result": result
    }
    
    # Switch to results page
    st.session_state.page = 'results'

def show_results():
    """Display the results of the video analysis"""
    results = st.session_state.results
    
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
    
    # Metrics section
    st.write("## Metrics and Analytics")
    
    # Check if we have bottleneck data
    if results["result"] and "bottlenecks" in results["result"]:
        # Show bottlenecks on a map
        st.write("### Bottleneck Analysis")
        
        bottlenecks = results["result"]["bottlenecks"]
        if bottlenecks:
            # Create a visualization of the bottlenecks
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot top view background if available
            if os.path.exists(results["top_view_video"]):
                # Get a frame from the top view video
                cap = cv2.VideoCapture(results["top_view_video"])
                ret, top_view_frame = cap.read()
                cap.release()
                
                if ret:
                    # Convert BGR to RGB
                    top_view_frame = cv2.cvtColor(top_view_frame, cv2.COLOR_BGR2RGB)
                    ax.imshow(top_view_frame)
            
            # Plot bottlenecks
            for i, bottleneck in enumerate(bottlenecks):
                ax.scatter(bottleneck["x"], bottleneck["y"], 
                          s=bottleneck["severity"] * 100, 
                          alpha=0.6, 
                          color='red')
                ax.annotate(f"Bottleneck {i+1}", 
                           (bottleneck["x"], bottleneck["y"]),
                           fontsize=12)
            
            ax.set_title("Bottleneck Locations and Severity")
            st.pyplot(fig)
            
            # Bottleneck details table
            st.write("### Bottleneck Details")
            bottleneck_data = []
            for i, bottleneck in enumerate(bottlenecks):
                bottleneck_data.append({
                    "ID": i+1,
                    "X": bottleneck["x"],
                    "Y": bottleneck["y"],
                    "Severity": bottleneck["severity"],
                    "Anomaly Count": bottleneck.get("anomaly_count", "N/A")
                })
            
            st.table(bottleneck_data)
    
    # Check if we have crowd density statistics
    if results["result"] and "crowd_stats" in results["result"]:
        st.write("### Crowd Density Statistics")
        
        crowd_stats = results["result"]["crowd_stats"]
        if crowd_stats:
            # Create a line chart of estimated count over time
            if "estimated_counts" in crowd_stats and crowd_stats["estimated_counts"]:
                counts = crowd_stats["estimated_counts"]
                frames = list(range(len(counts)))
                
                # Create the chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(frames, counts, '-o', markersize=4)
                ax.set_title("Estimated Crowd Count Over Time")
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Estimated Count")
                ax.grid(True)
                
                # Add peak labels
                peak_threshold = np.mean(counts) + np.std(counts)
                peaks = [(i, count) for i, count in enumerate(counts) if count > peak_threshold]
                
                for i, count in peaks:
                    ax.annotate(f"{count}", 
                               (i, count),
                               xytext=(0, 10),
                               textcoords="offset points",
                               fontsize=8)
                
                st.pyplot(fig)
            
            # Display overall statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Count", 
                         round(crowd_stats.get("average_count", 0), 1))
            
            with col2:
                st.metric("Max Count", 
                         crowd_stats.get("max_count", 0))
            
            with col3:
                st.metric("Anomalies Detected", 
                         crowd_stats.get("total_anomalies", 0))
            
            with col4:
                st.metric("Stampede Risk", 
                         f"{crowd_stats.get('stampede_risk', 0)}%")
    
    # Display raw results
    with st.expander("View Raw Processing Results"):
        st.json(results["result"])
    
    # Buttons to navigate or restart
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Edit Configuration"):
            st.session_state.page = 'configure'
    
    with col2:
        if st.button("Start New Analysis"):
            # Reset state and go back to upload page
            st.session_state.video_path = None
            st.session_state.frame = None
            st.session_state.perspective_points = None
            st.session_state.walking_areas = []
            st.session_state.roads = []
            st.session_state.current_area_points = []
            st.session_state.results = None
            st.session_state.page = 'upload'
            st.rerun()

def upload_page():
    """Video upload page"""
    st.title("BhedChaal - CCTV Crowd Analysis")
    st.write("### Upload Video for Analysis")
    
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file
        video_path = save_uploaded_file(uploaded_file)
        st.session_state.video_path = video_path
        st.write(f"Video uploaded successfully: {uploaded_file.name}")
        
        # Get the first frame for perspective selection
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            st.session_state.frame = frame
            st.session_state.page = 'perspective'
            st.rerun()
        else:
            st.error("Failed to read the uploaded video. Please try a different file.")

def perspective_page():
    """Page for selecting perspective points"""
    st.title("Step 1: Perspective Points")
    
    # Display perspective point selection interface
    perspective_points = select_perspective_points(st.session_state.frame)
    st.session_state.perspective_points = perspective_points
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go Back"):
            st.session_state.page = 'upload'
    
    with col2:
        if st.button("Continue to Area Selection"):
            st.session_state.page = 'areas'

def areas_page():
    """Page for defining walking areas and roads"""
    st.title("Step 2: Define Areas")
    
    # Display area selection interface
    walking_areas, roads = select_areas(st.session_state.frame)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go Back to Perspective Selection"):
            st.session_state.page = 'perspective'
    
    with col2:
        if st.button("Continue to Processing Options"):
            st.session_state.page = 'configure'

def configure_page():
    """Page for configuring processing options"""
    st.title("Step 3: Configure Processing Options")
    
    # Options form
    with st.form("processing_options"):
        st.write("### Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enhanced = st.checkbox("Enhanced Visualization", 
                                 value=st.session_state.processing_options["enhanced"])
            
            tracking = st.checkbox("Enable Tracking", 
                                 value=st.session_state.processing_options["tracking"])
            
            preprocess = st.checkbox("Preprocess Video", 
                                   value=st.session_state.processing_options["preprocess"])
        
        with col2:
            model_size = st.selectbox(
                "YOLOv8 Model Size",
                ["n", "s", "m", "l", "x"],
                index=["n", "s", "m", "l", "x"].index(st.session_state.processing_options["model_size"])
            )
        
        # Enhanced options
        if enhanced:
            st.write("### Enhanced Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                density_threshold = st.slider(
                    "Density Threshold",
                    min_value=0.1,
                    max_value=0.5,
                    value=st.session_state.processing_options["density_threshold"],
                    step=0.05
                )
                
                max_points = st.slider(
                    "Max Density Points",
                    min_value=50,
                    max_value=500,
                    value=st.session_state.processing_options["max_points"],
                    step=50
                )
            
            with col2:
                anomaly_threshold = st.slider(
                    "Anomaly Threshold",
                    min_value=10,
                    max_value=50,
                    value=st.session_state.processing_options["anomaly_threshold"],
                    step=5
                )
                
                stampede_threshold = st.slider(
                    "Stampede Threshold",
                    min_value=15,
                    max_value=55,
                    value=st.session_state.processing_options["stampede_threshold"],
                    step=5
                )
                
                max_bottlenecks = st.slider(
                    "Max Bottlenecks",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.processing_options["max_bottlenecks"],
                    step=1
                )
        else:
            # Set default values if enhanced is disabled
            density_threshold = st.session_state.processing_options["density_threshold"]
            max_points = st.session_state.processing_options["max_points"]
            anomaly_threshold = st.session_state.processing_options["anomaly_threshold"]
            stampede_threshold = st.session_state.processing_options["stampede_threshold"]
            max_bottlenecks = st.session_state.processing_options["max_bottlenecks"]
        
        submitted = st.form_submit_button("Save Configuration")
        
        if submitted:
            # Update processing options
            st.session_state.processing_options = {
                "enhanced": enhanced,
                "tracking": tracking,
                "model_size": model_size,
                "preprocess": preprocess,
                "density_threshold": density_threshold,
                "max_points": max_points,
                "anomaly_threshold": anomaly_threshold,
                "stampede_threshold": stampede_threshold,
                "max_bottlenecks": max_bottlenecks
            }
            st.success("Configuration saved!")
    
    # Process video button (outside the form)
    if st.button("Process Video"):
        if not st.session_state.walking_areas and not st.session_state.roads:
            st.warning("Warning: No walking areas or roads defined. This may affect the analysis.")
            if not st.button("Process Anyway"):
                return
        
        with st.spinner("Processing video... This may take a while."):
            # Run the video analysis
            run_video_analysis()

def results_page():
    """Page for displaying results"""
    st.title("Analysis Results")
    
    if st.session_state.results:
        show_results()
    else:
        st.error("No results available. Please process a video first.")
        if st.button("Go to Upload"):
            st.session_state.page = 'upload'

def main():
    # Handle page routing
    if st.session_state.page == 'upload':
        upload_page()
    elif st.session_state.page == 'perspective':
        perspective_page()
    elif st.session_state.page == 'areas':
        areas_page()
    elif st.session_state.page == 'configure':
        configure_page()
    elif st.session_state.page == 'results':
        results_page()

if __name__ == "__main__":
    main() 