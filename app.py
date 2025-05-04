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

# Import from existing codebase
from visualization import process_cctv_to_top_view, get_perspective_transform
from simulation_enhancement import enhanced_process_cctv_to_top_view
from person_detection import AreaManager, PersonDetector
from density_estimation import CrowdDensityEstimator
from anomaly_detection import AnomalyDetector
from video_preprocessor import preprocess_video
from panic_simulation import PanicSimulation

# Set page config
st.set_page_config(
    page_title="BhedChaal - CCTV Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define global variables
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return the file path"""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def select_perspective_points(image):
    """Streamlit interface to select 4 points for perspective transformation"""
    st.write("### Select 4 points for perspective transformation")
    st.write("Click on 4 points in the image that form a rectangle in the real world.")
    st.write("Select in clockwise order: top-left, top-right, bottom-right, bottom-left")
    
    # Display the image for point selection
    h, w = image.shape[:2]
    
    # Calculate display width maintaining aspect ratio
    display_width = 800
    display_height = int(h * display_width / w)
    
    # Create a canvas for point selection
    canvas_result = st.image(image, width=display_width, channels="BGR")
    
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

def run_panic_simulation(video_path, auto_record=False, record_duration=60):
    """Run panic simulation using the video data"""
    try:
        st.write("### Launching Panic Simulation")
        status_placeholder = st.empty()
        status_placeholder.info("Initializing panic simulation...")
        
        # Create simulation instance
        simulation = PanicSimulation()
        
        # Load data
        status_placeholder.info("Loading data from video...")
        if simulation.load_data_from_video(video_path):
            # Auto-start recording if requested
            if auto_record:
                status_placeholder.info(f"Auto-recording enabled for {record_duration} seconds")
                simulation.start_recording(duration=record_duration)
            
            # Run simulation
            status_placeholder.info("Running panic simulation... (check for a new Pygame window)")
            simulation.run_simulation()
            status_placeholder.success("Panic simulation completed.")
            
            # If there are recorded results, show them
            results_dir = "panic_sim_results"
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                if result_files:
                    st.write("### Panic Simulation Results")
                    for file in result_files:
                        st.video(os.path.join(results_dir, file))
            
            return True
        else:
            status_placeholder.error(f"Failed to load data from {video_path}")
            return False
    except Exception as e:
        st.error(f"Error running panic simulation: {e}")
        return False

def show_results(results, video_path):
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
    
    # Panic simulation section
    st.write("## Panic Simulation")
    st.write("Simulate potential stampede scenarios based on the crowd analysis.")
    
    # Panic simulation options
    panic_col1, panic_col2 = st.columns(2)
    
    with panic_col1:
        auto_record = st.checkbox("Auto-record simulation", value=True)
    
    with panic_col2:
        record_duration = st.slider("Recording duration (seconds)", 10, 120, 60)
    
    if st.button("Launch Panic Simulation"):
        run_panic_simulation(video_path, auto_record, record_duration)

def main():
    st.title("BhedChaal - CCTV Crowd Analysis")
    
    # Sidebar for options
    st.sidebar.title("Options")
    
    # Video input
    st.sidebar.header("Input")
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    # Processing options
    st.sidebar.header("Processing Options")
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
    
    # Panic Simulation Options
    st.sidebar.header("Panic Simulation Options")
    run_panic_sim = st.sidebar.checkbox("Launch Panic Simulation After Processing", value=False)
    panic_auto_record = st.sidebar.checkbox("Auto-record Panic Simulation", value=True)
    panic_record_duration = st.sidebar.slider(
        "Recording Duration (seconds)",
        min_value=10,
        max_value=120,
        value=60,
        step=10
    )
    
    # Main area
    if uploaded_file is not None:
        # Save uploaded file
        video_path = save_uploaded_file(uploaded_file)
        st.write(f"Video uploaded successfully: {uploaded_file.name}")
        
        # Get the first frame for perspective selection
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Display options for perspective transformation
            src_points = select_perspective_points(frame)
            
            # Process button
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
                    results = run_video_analysis(video_path, src_points, options)
                    
                    # Show results
                    show_results(results, video_path)
                    
                    # Automatically launch panic simulation if option is selected
                    if run_panic_sim:
                        run_panic_simulation(video_path, panic_auto_record, panic_record_duration)
        else:
            st.error("Failed to read the uploaded video. Please try a different file.")
    else:
        st.info("Please upload a video file to begin analysis.")

if __name__ == "__main__":
    main() 