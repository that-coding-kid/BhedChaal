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
from streamlit_drawable_canvas import st_canvas

# Import from existing codebase
from visualization import process_cctv_to_top_view, get_perspective_transform
from simulation_enhancement import enhanced_process_cctv_to_top_view
from person_detection import AreaManager, PersonDetector
from density_estimation import CrowdDensityEstimator
from anomaly_detection import AnomalyDetector
from video_preprocessor import preprocess_video

# Set page config
st.set_page_config(
    page_title="BhedChaal Interactive - CCTV Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define global variables
TEMP_DIR = "temp"
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
if 'canvas_result' not in st.session_state:
    st.session_state.canvas_result = None
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'results' not in st.session_state:
    st.session_state.results = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the file path"""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def display_interactive_point_selection(frame):
    """
    Display interactive canvas for point selection
    """
    st.write("### Select 4 points for perspective transformation")
    st.write("Click to place points on the image in clockwise order: top-left, top-right, bottom-right, bottom-left")
    
    # Get image dimensions
    h, w = frame.shape[:2]
    
    # Calculate display width maintaining aspect ratio
    display_width = 800
    display_height = int(h * display_width / w)
    
    # Convert CV2 BGR image to RGB for display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create two columns - one for the canvas, one for the instructions/buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create the canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fill color for circle
            stroke_width=2,
            stroke_color="#00FF00",  # Green border for circle
            background_image=Image.fromarray(rgb_frame),
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="point",
            point_display_radius=5,
            key="perspective_canvas",
        )
    
    with col2:
        st.write("### Point Selection")
        st.write("Click on the image to select points")
        
        # Display current points
        if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
            points = []
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "point":
                    # Convert from canvas coordinates to original image coordinates
                    x = int(obj["left"] * w / display_width)
                    y = int(obj["top"] * h / display_height)
                    points.append([x, y])
            
            # Update session state
            st.session_state.points = points
            
            # Display the points
            if len(points) > 0:
                st.write("Selected points:")
                for i, point in enumerate(points):
                    st.write(f"Point {i+1}: ({point[0]}, {point[1]})")
        
        # Button to clear points
        if st.button("Clear Points"):
            st.session_state.points = []
            st.rerun()
        
        # Button to confirm points
        if st.button("Confirm Points") and len(st.session_state.points) >= 4:
            # Take the first 4 points
            st.session_state.perspective_points = st.session_state.points[:4]
            st.session_state.step = "options"
            st.rerun()
    
    # Create a preview with the current points
    if st.session_state.points:
        preview_img = frame.copy()
        for i, point in enumerate(st.session_state.points[:4]):  # Limit to first 4 points
            # Draw points on image
            cv2.circle(preview_img, tuple(point), 5, (0, 255, 0), -1)
            
            # Draw lines connecting points
            if i > 0:
                cv2.line(preview_img, tuple(st.session_state.points[i-1]), tuple(point), (0, 255, 0), 2)
            
            # Connect last point to first if we have 4 points
            if i == 3:
                cv2.line(preview_img, tuple(point), tuple(st.session_state.points[0]), (0, 255, 0), 2)
        
        # Display the preview
        st.image(preview_img, caption="Preview with selected points", width=display_width, channels="BGR")
    
    return st.session_state.points

def run_video_analysis(video_path, src_points, options):
    """Run video analysis with the existing code and return results"""
    
    # Create output paths
    video_name = Path(video_path).stem
    output_dir = os.path.join(TEMP_DIR, "output_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    output_original = os.path.join(output_dir, f"{video_name}_enhanced_original.mp4")
    output_top_view = os.path.join(output_dir, f"{video_name}_enhanced_top_view.mp4")
    output_density = os.path.join(output_dir, f"{video_name}_enhanced_density.mp4")
    
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
            max_bottlenecks=options["max_bottlenecks"]
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
            max_bottlenecks=options["max_bottlenecks"]
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
        st.rerun()

def main():
    st.title("BhedChaal - Interactive CCTV Crowd Analysis")
    
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
            
            # Get the first frame for perspective selection
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                st.session_state.frame = frame
                st.session_state.step = "perspective"
                st.rerun()
            else:
                st.error("Failed to read the uploaded video. Please try a different file.")
    
    # Perspective selection step
    elif st.session_state.step == "perspective":
        if st.session_state.frame is not None:
            display_interactive_point_selection(st.session_state.frame)
            
            # Button to go back to upload
            if st.button("Back to Upload"):
                st.session_state.step = "upload"
                st.rerun()
    
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
        
        # Process button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Point Selection"):
                st.session_state.step = "perspective"
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
                        "max_bottlenecks": max_bottlenecks
                    }
                    
                    # Run analysis
                    results = run_video_analysis(st.session_state.video_path, st.session_state.perspective_points, options)
                    
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