import cv2
import numpy as np
import time
import os
import json
from pathlib import Path
from density_estimation import CrowdDensityEstimator
from person_detection import PersonDetector
from person_detection import AreaManager
from anomaly_detection import AnomalyDetector

def get_perspective_transform(image, src_points=None, dst_size=(800, 600), area_manager=None):
    """
    Get perspective transformation matrix from user-selected points or load from saved data
    
    Parameters:
    - image: Source image
    - src_points: Four points in the source image (if None, prompt user to select)
    - dst_size: Size of the destination (top-view) image
    - area_manager: Instance of AreaManager to save/load perspective points
    
    Returns:
    - perspective_matrix: Homography matrix for perspective transformation
    - inv_perspective_matrix: Inverse homography matrix
    - src_points: The source points used (loaded or selected)
    """
    # First try to load perspective points if area_manager is provided
    loaded_points = False
    if area_manager is not None:
        try:
            if area_manager.load_perspective_points():
                src_points = area_manager.perspective_points
                loaded_points = True
                print("Loaded perspective points from saved data")
        except Exception as e:
            print(f"Error loading perspective points: {e}")
    
    # If src_points is None or loading failed, let user select 4 points
    if src_points is None or len(src_points) != 4:
        # Let user select 4 points
        src_points = []
        img_copy = image.copy()
        
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                src_points.append([x, y])
                # Draw point
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                if len(src_points) > 1:
                    # Connect points
                    cv2.line(img_copy, tuple(src_points[-2]), (x, y), (0, 255, 0), 2)
                if len(src_points) == 4:
                    # Connect last point to first
                    cv2.line(img_copy, tuple(src_points[0]), tuple(src_points[3]), (0, 255, 0), 2)
                cv2.imshow('Select 4 points (clockwise from top-left)', img_copy)
        
        print("Please select 4 points that form a rectangle in the real world.")
        print("Select in clockwise order: top-left, top-right, bottom-right, bottom-left")
        
        cv2.imshow('Select 4 points (clockwise from top-left)', img_copy)
        cv2.setMouseCallback('Select 4 points (clockwise from top-left)', click_event)
        
        # Wait until 4 points are selected
        while len(src_points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.waitKey(500)  # Small delay
        cv2.destroyAllWindows()
        
        # Save the perspective points if area_manager is provided
        if area_manager is not None:
            area_manager.save_perspective_points(src_points)
            print("Saved perspective points")
    
    # Destination points (top-view rectangle)
    margin = 50  # margin from the edges
    dst_width, dst_height = dst_size
    dst_points = np.array([
        [margin, margin],  # Top-left
        [dst_width - margin, margin],  # Top-right
        [dst_width - margin, dst_height - margin],  # Bottom-right
        [margin, dst_height - margin]  # Bottom-left
    ], dtype=np.float32)
    
    # Convert source points to numpy array
    src_points = np.array(src_points, dtype=np.float32)
    
    # Compute the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inv_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    return perspective_matrix, inv_perspective_matrix, src_points

def transform_point(point, matrix):
    """Transform a point using the given homography matrix"""
    # Convert to homogeneous coordinates
    homogeneous_point = np.array([point[0], point[1], 1.0])
    
    # Apply transformation
    transformed = matrix.dot(homogeneous_point)
    
    # Convert back from homogeneous coordinates
    transformed /= transformed[2]
    
    return (int(transformed[0]), int(transformed[1]))

def transform_polygon(polygon, matrix):
    """Transform a polygon using the given homography matrix"""
    transformed_polygon = []
    for point in polygon:
        transformed_point = transform_point(point, matrix)
        transformed_polygon.append(transformed_point)
    return np.array(transformed_polygon)

def create_top_view(frame, density_map, person_detector, homography, area_manager, estimated_count, 
                    size=(800, 600), frame_number=None, save_data=False):
    """
    Create a top view visualization with transformed density map and tracking
    
    Parameters:
    - frame: Input frame
    - density_map: Density map from crowd estimator
    - person_detector: Instance of PersonDetector
    - homography: Perspective transformation matrix
    - area_manager: Instance of AreaManager
    - estimated_count: Estimated people count
    - size: Size of the top view image (width, height)
    - frame_number: Current frame number for saving data
    - save_data: Whether to save object and density point data
    
    Returns:
    - top_view: Top view visualization
    """
    # Create a blank top-view image
    top_view = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Draw a grid for reference
    for x in range(0, size[0], 50):
        cv2.line(top_view, (x, 0), (x, size[1]), (200, 200, 200), 1)
    for y in range(0, size[1], 50):
        cv2.line(top_view, (0, y), (size[0], y), (200, 200, 200), 1)
    
    # Transform walking areas to top view
    top_view_walking_areas = []
    for area in area_manager.walking_areas:
        top_view_area = transform_polygon(area, homography)
        top_view_walking_areas.append(top_view_area)
        cv2.fillPoly(top_view, [top_view_area], (0, 255, 0, 128))
        cv2.polylines(top_view, [top_view_area], True, (0, 255, 0), 2)
    
    # Transform roads to top view
    top_view_roads = []
    for road in area_manager.roads:
        top_view_road = transform_polygon(road, homography)
        top_view_roads.append(top_view_road)
        cv2.fillPoly(top_view, [top_view_road], (0, 0, 255, 128))
        cv2.polylines(top_view, [top_view_road], True, (0, 0, 255), 2)
    
    # Transform density map to top view if available
    density_points_data = []
    if density_map is not None:
        # Create a colorized version of the density map for visualization
        norm_density = density_map / (np.max(density_map) + 1e-10)
        colorized_density = cv2.applyColorMap((norm_density * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Transform the colorized density map to top view
        warped_density = cv2.warpPerspective(colorized_density, homography, size)
        
        # Apply threshold to density map
        # Create mask where density is above threshold
        mask = cv2.warpPerspective((norm_density > 0.1).astype(np.uint8) * 255, homography, size)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to warped density
        warped_density = cv2.bitwise_and(warped_density, mask)
        
        # Blend the warped density map with top view
        alpha = 0.6
        top_view = cv2.addWeighted(top_view, 1-alpha, warped_density, alpha, 0)
        
        # NEW: Extract density points for saving
        if save_data and frame_number is not None:
            # Find areas with significant density
            high_density_areas = norm_density > 0.2  # Threshold value
            if np.any(high_density_areas):
                y_coords, x_coords = np.where(high_density_areas)
                densities = norm_density[high_density_areas]
                
                # Create list of top view coordinates with density values
                for i, (x, y, density_value) in enumerate(zip(x_coords, y_coords, densities)):
                    # Limit to 100 points to avoid excessive data
                    if i >= 100:
                        break
                        
                    # Transform point to top view
                    top_view_point = transform_point((int(x), int(y)), homography)
                    
                    # Add to density points data
                    density_points_data.append({
                        "top_view": [top_view_point[0], top_view_point[1]],
                        "original": [int(x), int(y)],
                        "density_value": float(density_value)
                    })
    
    # Transform tracked people to top view and represent them as dots with IDs
    # First, get the current active IDs (people currently detected in the frame)
    current_ids = []
    object_data = []
    detections, _ = person_detector.detect(frame)
    
    for detection in detections:
        if len(detection) >= 6 and detection[5] is not None:
            current_ids.append(detection[5])
    
    if person_detector.tracker_enabled:
        # First, draw the tracking lines for all historical tracks
        for track_id, track in person_detector.track_history.items():
            if len(track) > 1:
                # Transform track points to top view
                top_view_track = []
                for point in track:
                    top_view_point = transform_point(point, homography)
                    top_view_track.append(top_view_point)
                
                # Draw the track line with the same color as original
                color = person_detector._get_color_by_id(track_id)
                for i in range(1, len(top_view_track)):
                    # Use faded color for tracks that aren't currently visible
                    line_color = color if track_id in current_ids else (200, 200, 200)
                    cv2.line(top_view, top_view_track[i-1], top_view_track[i], line_color, 2)
                
                # Only draw dots and IDs for currently detected people
                if track_id in current_ids and len(top_view_track) > 0:
                    current_pos = top_view_track[-1]
                    # Draw a larger filled circle for the person
                    cv2.circle(top_view, current_pos, 8, color, -1)
                    # Draw ID number next to the dot
                    cv2.putText(top_view, str(track_id), 
                                (current_pos[0] + 10, current_pos[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else:
        # If tracking is not enabled, still use the ID from the detector if available
        # This uses the detection information we already collected above
        for detection in detections:
            # Check if we have ID information
            if len(detection) >= 6:
                x1, y1, x2, y2, confidence, track_id = detection
            else:
                x1, y1, x2, y2, confidence = detection
                track_id = None
                
            # Calculate center point of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Transform center point to top view
            top_view_point = transform_point((center_x, center_y), homography)
            
            # Choose color based on ID if available
            if track_id is not None:
                color = person_detector._get_color_by_id(track_id)
                id_text = str(track_id)
            else:
                # Use a default color scheme and sequential numbering
                color = (0, 0, 255)  # Red for detections without IDs
                # Get the index in the detections list
                idx = detections.index(detection)
                id_text = f"D{idx}"  # prefix with 'D' to indicate it's not a tracked ID
            
            # Draw a dot for the detection
            cv2.circle(top_view, top_view_point, 8, color, -1)
            
            # Draw ID next to the dot
            cv2.putText(top_view, id_text, 
                        (top_view_point[0] + 10, top_view_point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # NEW: Save object data if requested
            if save_data and frame_number is not None:
                object_data.append({
                    "id": track_id if track_id is not None else f"D{idx}",
                    "orig_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "top_view": [top_view_point[0], top_view_point[1]],
                    "confidence": float(confidence)
                })
    
    # Add legend
    cv2.rectangle(top_view, (size[0]-240, 10), (size[0]-10, 150), (255, 255, 255), -1)
    cv2.rectangle(top_view, (size[0]-240, 10), (size[0]-10, 150), (0, 0, 0), 1)
    cv2.putText(top_view, "Legend:", (size[0]-230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Walking area
    cv2.rectangle(top_view, (size[0]-220, 50), (size[0]-200, 60), (0, 255, 0), -1)
    cv2.putText(top_view, "Walking Area", (size[0]-190, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Road
    cv2.rectangle(top_view, (size[0]-220, 70), (size[0]-200, 80), (0, 0, 255), -1)
    cv2.putText(top_view, "Road", (size[0]-190, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Active person
    cv2.circle(top_view, (size[0]-210, 100), 7, (0, 0, 255), -1)
    cv2.putText(top_view, "Active Person", (size[0]-190, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Historical track
    cv2.line(top_view, (size[0]-220, 120), (size[0]-200, 120), (200, 200, 200), 2)
    cv2.putText(top_view, "Historical Track", (size[0]-190, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Current count
    num_active_people = len([d for d in detections if len(d) >= 6 and d[5] is not None])
    cv2.putText(top_view, f"Active People: {num_active_people}", (10, size[1] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add density count information
    cv2.putText(top_view, f"Estimated people count: {estimated_count*0.05:.1f}", (10, size[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save data if requested
    if save_data and frame_number is not None and area_manager is not None:
        # Save detected objects
        if object_data:
            area_manager.save_detected_objects(frame_number, object_data)
        
        # Save density points
        if density_points_data:
            area_manager.save_density_points(frame_number, density_points_data)
    
    return top_view

def process_cctv_to_top_view(video_path, output_path=None, calibration_image=None, src_points=None, 
                             use_tracking=True, yolo_model_size='x', csrnet_model_path=None,
                             save_data=True, load_saved_data=True, preprocess_video=True,
                             anomaly_threshold=30, stampede_threshold=35, max_bottlenecks=3):
    """
    Process CCTV footage to create a top-view simulation with crowd density estimation and YOLOv8 person detection
    
    Parameters:
    - video_path: Path to input CCTV video
    - output_path: Path for output video (if None, don't save)
    - calibration_image: Path to image for calibration (if None, use first frame)
    - src_points: Four corner points in the source image (if None, prompt user or load from saved data)
    - use_tracking: Whether to enable person tracking
    - yolo_model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x')
    - csrnet_model_path: Path to CSRNet pre-trained weights (if None, use default)
    - save_data: Whether to save area, perspective points, and detection data
    - load_saved_data: Whether to load previously saved data
    - preprocess_video: Whether to preprocess the video to standardize resolution
    - anomaly_threshold: Threshold to identify bottlenecks when anomalies exceed this value
    - stampede_threshold: Threshold to trigger stampede warning when anomalies exceed this value
    - max_bottlenecks: Maximum number of bottlenecks to identify (default: 3)
    """
    # Preprocess video if requested to standardize resolution
    original_video_path = video_path
    processed_video = None
    
    if preprocess_video:
        try:
            from video_preprocessor import VideoPreprocessor
            preprocessor = VideoPreprocessor(target_resolution=(1280, 720))
            print(f"Preprocessing video: {video_path}")
            processed_video = preprocessor.process_video(video_path)
            if processed_video != video_path:
                print(f"Video preprocessed to standardized resolution: {processed_video}")
                video_path = processed_video
        except Exception as e:
            print(f"Warning: Video preprocessing failed ({str(e)}). Using original video.")
    
    # Initialize area manager with current video path for data management
    area_manager = AreaManager(video_path=original_video_path, save_dir="video_data")
    
    # Initialize crowd density estimator
    print("Setting up Crowd Density Estimator...")
    try:
        crowd_estimator = CrowdDensityEstimator(model_path=csrnet_model_path)
    except Exception as e:
        print(f"Error setting up Crowd Density Estimator: {e}")
        print("Continuing without crowd density estimation.")
        crowd_estimator = None
    
    # Initialize YOLOv8 person detector
    print("Setting up YOLOv8 person detector...")
    try:
        person_detector = PersonDetector(model_size=yolo_model_size)
        # Tracking is now enabled by default in the updated PersonDetector
        # but can be disabled if needed
        if not use_tracking:
            person_detector.enable_tracking(False)
    except Exception as e:
        print(f"Error setting up YOLOv8 person detector: {e}")
        print("Please make sure you have the ultralytics package installed and YOLOv8 weights available.")
        return None
    
    # Initialize anomaly detector for counter-flow detection
    print("Setting up Anomaly Detector...")
    anomaly_detector = AnomalyDetector(angle_threshold=65, history_length=5, 
                                      anomaly_persistence=60, anomaly_threshold=anomaly_threshold,
                                      stampede_threshold=stampede_threshold,
                                      max_bottlenecks=max_bottlenecks)
    print(f"Anomaly threshold for bottleneck detection: {anomaly_threshold}")
    print(f"Stampede warning threshold: {stampede_threshold}")
    print(f"Maximum number of bottlenecks: {max_bottlenecks}")
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If load_saved_data is True, try to load areas and perspective points first
    perspective_points_loaded = False
    areas_loaded = False
    
    if load_saved_data:
        # Try to load areas
        if area_manager.load_areas():
            areas_loaded = True
            print("Successfully loaded areas from saved data")
        
        # Try to load perspective points
        if area_manager.load_perspective_points():
            perspective_points_loaded = True
            print("Successfully loaded perspective points from saved data")
    
    # If no calibration image provided, use the first frame
    if calibration_image is None:
        print("Extracting first frame for calibration...")
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the first frame")
        calibration_image = "first_frame.jpg"
        cv2.imwrite(calibration_image, first_frame)
        
        # Define areas if they weren't loaded
        if not areas_loaded:
            print("Please define walking areas and roads on the frame...")
            area_manager.define_areas(first_frame)
        
        # Get perspective transformation matrix
        print("Getting perspective transformation matrix...")
        top_view_size = (800, 600)
        
        if perspective_points_loaded:
            # If perspective points were loaded, use them
            src_points = area_manager.perspective_points
        
        homography, inv_homography, src_points = get_perspective_transform(
            first_frame, src_points, top_view_size, area_manager if save_data else None
        )
        
        # Reset video to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # If calibration image is provided, use it
        calibration_frame = cv2.imread(calibration_image)
        if calibration_frame is None:
            raise ValueError(f"Could not read calibration image {calibration_image}")
        
        # Define areas if they weren't loaded
        if not areas_loaded:
            print("Please define walking areas and roads on the calibration image...")
            area_manager.define_areas(calibration_frame)
        
        # Get perspective transformation matrix
        print("Getting perspective transformation matrix...")
        top_view_size = (800, 600)
        
        if perspective_points_loaded:
            # If perspective points were loaded, use them
            src_points = area_manager.perspective_points
        
        homography, inv_homography, src_points = get_perspective_transform(
            calibration_frame, src_points, top_view_size, area_manager if save_data else None
        )
    
    # Setup video writers if output path is provided
    out_top_view = None
    out_original = None
    out_density = None

    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get source video filename without extension to use as prefix
        source_video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Base filename without extension
        if output_path == None:
            base_path = source_video_name
        else:
            base_path = os.path.splitext(output_path)[0]
        
        # Create different output files for each view with source video prefix
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_top_view = cv2.VideoWriter(f"{base_path}_{source_video_name}_top_view.mp4", fourcc, fps, top_view_size)
        out_original = cv2.VideoWriter(f"{base_path}_{source_video_name}_original.mp4", fourcc, fps, (width, height))
        
        # Only create density output if we have a density estimator
        if crowd_estimator is not None:
            out_density = cv2.VideoWriter(f"{base_path}_{source_video_name}_density.mp4", fourcc, fps, (width, height))
        
        print(f"Output videos will be saved with prefix: {source_video_name}")

    # Create directories for saving snapshots
    snapshots_dir = None
    if output_path:
        # Use the source video name in the snapshots directory name
        source_video_name = os.path.splitext(os.path.basename(video_path))[0]
        base_path = os.path.splitext(output_path)[0]
        snapshots_dir = f"{base_path}_{source_video_name}_snapshots"
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir)
            os.makedirs(os.path.join(snapshots_dir, "top_view"))
            os.makedirs(os.path.join(snapshots_dir, "original"))
            if crowd_estimator is not None:
                os.makedirs(os.path.join(snapshots_dir, "density"))
        print(f"Snapshots will be saved to {snapshots_dir} directory")
    
    # Interval for saving snapshots (every X frames)
    snapshot_interval = 30  # Save every 30 frames (adjust as needed)
    
    # Process video
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")
    
    # Print instructions for the user
    print("Controls:")
    print("  'q' - Quit the application")
    print("  'r' - Reset heat maps")
    print("  't' - Toggle tracking visualization (on/off)")
    print("  's' - Toggle data saving (on/off)")
    print("\nPoints in the top view only appear for currently detected people.")
    print("Historical tracks are shown in gray, active tracks in color.")
    print("\nNote: When data saving is ON, data is saved every 20 frames to reduce disk usage.")
    
    # For FPS calculation
    start_time = time.time()
    frame_time = start_time
    
    # Create windows with fixed positions for better visualization
    cv2.namedWindow('Original with Density and Detections', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Original with Density and Detections', 50, 50)
    cv2.resizeWindow('Original with Density and Detections', 640, 480)
    
    cv2.namedWindow('Top View', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Top View', 700, 50)
    cv2.resizeWindow('Top View', 800, 600)
    
    if crowd_estimator is not None:
        cv2.namedWindow('Crowd Density Heat Map', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Crowd Density Heat Map', 50, 550)
        cv2.resizeWindow('Crowd Density Heat Map', 640, 480)
    
    # Toggle for data saving (can be toggled during processing)
    current_save_data = save_data
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - frame_time
        frame_time = current_time
        fps_current = 1.0 / elapsed if elapsed > 0 else 0
        
        # First, check if we have saved detection data for this frame
        saved_objects = None
        saved_density_points = None
        
        if load_saved_data:
            saved_objects = area_manager.load_detected_objects(frame_count)
            if crowd_estimator is not None:
                saved_density_points = area_manager.load_density_points(frame_count)
        
        # If we have saved data for this frame, we can skip detection and use saved data
        # However, we still need to process the frame for visualization purposes
        
        # Estimate crowd density (if available)
        density_map = None
        estimated_count = 0
        colorized_density = None
        
        if crowd_estimator is not None:
            try:
                density_map, estimated_count, colorized_density = crowd_estimator.estimate_density(frame)
            except Exception as e:
                print(f"Error in density estimation: {e}")
                # Continue without density estimation for this frame
        
        # Detect and track individual people with YOLOv8
        detections, frame_with_detections = person_detector.detect(frame)
        
        # Create a combined visualization
        frame_with_visualization = frame_with_detections.copy()
        
        # 1. Add density visualization if available
        if colorized_density is not None:
            # Blend with original frame
            alpha = 0.6
            mask = (density_map > 0.1).astype(np.uint8) * 255
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
            masked_density = cv2.bitwise_and(colorized_density, mask)
            cv2.addWeighted(frame_with_visualization, 1.0, masked_density, alpha, 0, frame_with_visualization)
        
        # 2. Draw areas on the combined frame
        area_manager.draw_on_frame(frame_with_visualization)
        
        # 3. Create top view with density map and tracking
        # Only save data every 20 frames to reduce disk usage
        save_current_frame = current_save_data and (frame_count % 20 == 0)
        
        top_view = create_top_view(
            frame, 
            density_map, 
            person_detector, 
            homography, 
            area_manager, 
            estimated_count, 
            top_view_size,
            frame_count if save_current_frame else None,
            save_current_frame
        )
        
        # Add detection information to original view
        # Now includes the number of people with IDs
        person_count_with_ids = sum(1 for det in detections if len(det) >= 6 and det[5] is not None)
        cv2.putText(frame_with_visualization, f"YOLO11 detections: {len(detections)}", (10, height - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_with_visualization, f"People with IDs: {person_count_with_ids}", (10, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_with_visualization, f"Density estimate: {estimated_count*0.05:.1f}", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add data saving indicator
        save_text = "Data Saving: ON" if current_save_data else "Data Saving: OFF"
        cv2.putText(frame_with_visualization, save_text, (width - 200, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if current_save_data else (0, 0, 255), 2)
        
        # Add FPS display
        cv2.putText(frame_with_visualization, f"FPS: {fps_current:.1f}", (width - 150, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frames
        cv2.imshow('Original with Density and Detections', frame_with_visualization)
        
        # Display separate density map if available
        if colorized_density is not None:
            cv2.imshow('Crowd Density Heat Map', colorized_density)
        
        # Display top view
        cv2.imshow('Top View', top_view)
        
        # Write frames to output videos
        if out_top_view:
            out_top_view.write(top_view)
        
        if out_original:
            out_original.write(frame_with_visualization)
            
        if out_density and colorized_density is not None:
            # Ensure density map is the right size and format for video
            if colorized_density.shape[:2] != (height, width):
                colorized_density = cv2.resize(colorized_density, (width, height))
            out_density.write(colorized_density)
        
        # Exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset heat maps if 'r' is pressed
            if crowd_estimator is not None:
                crowd_estimator.reset_heat_map()
            print("Heat maps reset")
        elif key == ord('t'):
            # Toggle tracking on/off if 't' is pressed
            use_tracking = not use_tracking
            person_detector.enable_tracking(use_tracking)
            print(f"Tracking {'enabled' if use_tracking else 'disabled'}")
        elif key == ord('s'):
            # Toggle data saving on/off
            current_save_data = not current_save_data
            print(f"Data saving {'enabled' if current_save_data else 'disabled'}")
        
        # Save snapshots at regular intervals
        if snapshots_dir and frame_count % snapshot_interval == 0:
            timestamp = time.strftime("%H%M%S")
            # Save top view
            cv2.imwrite(
                os.path.join(snapshots_dir, "top_view", f"frame_{frame_count:06d}_{timestamp}.jpg"), 
                top_view
            )
            # Save original view
            cv2.imwrite(
                os.path.join(snapshots_dir, "original", f"frame_{frame_count:06d}_{timestamp}.jpg"), 
                frame_with_visualization
            )
            # Save density map if available
            if colorized_density is not None:
                cv2.imwrite(
                    os.path.join(snapshots_dir, "density", f"frame_{frame_count:06d}_{timestamp}.jpg"), 
                    colorized_density
                )
            
            # Optional: Save the raw density data as NumPy array for further analysis
            if density_map is not None:
                np.save(
                    os.path.join(snapshots_dir, "density", f"raw_density_{frame_count:06d}_{timestamp}.npy"), 
                    density_map
                )
        
        frame_count += 1
    
    # Release resources
    cap.release()
    
    # Close all video writers
    if out_top_view:
        out_top_view.release()
    if out_original:
        out_original.release()
    if out_density:
        out_density.release()
        
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.2f} seconds.")
    
    if output_path:
        base_path = os.path.splitext(output_path)[0]
        print(f"Output saved to:")
        print(f"  - {base_path}_top_view.mp4 (Top view with tracking)")
        print(f"  - {base_path}_original.mp4 (Original view with detections)")
        if crowd_estimator is not None:
            print(f"  - {base_path}_density.mp4 (Crowd density heat map)")
    
    # Save a final frame of each visualization as an image
    if output_path and frame_count > 0:
        cv2.imwrite(f"{base_path}_top_view_final.jpg", top_view)
        cv2.imwrite(f"{base_path}_original_final.jpg", frame_with_visualization)
        if colorized_density is not None:
            cv2.imwrite(f"{base_path}_density_final.jpg", colorized_density)
        print(f"Final frames saved as JPG images.")
        
    # Also print how many frames had data saved
    if save_data:
        object_count = len(os.listdir(os.path.join(area_manager.objects_dir)))
        density_count = len(os.listdir(os.path.join(area_manager.density_dir)))
        print(f"Data saved for {object_count} object detection frames and {density_count} density frames (every 20th frame).")
    
    # Clean up processed video if it was created
    if preprocess_video and processed_video is not None and processed_video != original_video_path:
        try:
            from video_preprocessor import VideoPreprocessor
            preprocessor = VideoPreprocessor()
            preprocessor.cleanup()
            print("Cleaned up temporary processed videos")
        except Exception as e:
            print(f"Warning: Failed to clean up processed video: {str(e)}")
        
    return base_path + "_top_view.mp4" if output_path else None