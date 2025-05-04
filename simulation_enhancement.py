import numpy as np
import random
from collections import defaultdict
import os
import time
from density_estimation import CrowdDensityEstimator
from person_detection import PersonDetector
from visualization import AreaManager
from visualization import get_perspective_transform
from visualization import create_top_view, transform_point, transform_polygon
from anomaly_detection import AnomalyDetector

import cv2
import numpy as np
import random
from collections import defaultdict
import colorsys
import argparse
import sys

def create_enhanced_top_view(frame, density_map, previous_density_map, person_detector, homography, area_manager, 
                            estimated_count, size=(800, 600), 
                            density_threshold=0.1, max_density_points=100, show_flow=True,
                            frame_number=None, save_data=False, anomaly_detector=None):
    """
    Create an enhanced top view visualization with transformed density map, tracking,
    and additional density points based on crowd estimation
    
    Parameters:
    - frame: Input frame
    - density_map: Density map from crowd estimator
    - previous_density_map: Previous frame's density map for flow calculation
    - person_detector: Instance of PersonDetector
    - homography: Perspective transformation matrix
    - area_manager: Instance of AreaManager
    - estimated_count: Estimated people count
    - size: Size of the top view image (width, height)
    - density_threshold: Threshold for showing density points (0.0-1.0)
    - max_density_points: Maximum number of density points to display
    - show_flow: Whether to show flow visualization
    - frame_number: Current frame number for saving data
    - save_data: Whether to save object and density point data
    - anomaly_detector: Instance of AnomalyDetector for counter-flow detection
    
    Returns:
    - top_view: Enhanced top view visualization
    """
    # Create a blank top-view image (same as original function)
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
    
    # New enhanced visualization: Transform density map to top view and generate points
    density_points = []
    density_points_data = []  # For saving to JSON
    warped_density = None  # Store warped density for anomaly detection
    
    if density_map is not None:
        # Create a colorized version of the density map for visualization background
        norm_density = density_map / (np.max(density_map) + 1e-10)
        
        # Use a custom colormap with high contrast
        heat_map = np.zeros((256, 1, 3), dtype=np.uint8)
        # Create gradient from dark blue to cyan to bright green to yellow
        for i in range(256):
            if i < 64:  # Dark blue to cyan
                heat_map[i] = [255, i*4, 0]  # Increasing green component
            elif i < 128:  # Cyan to bright green
                heat_map[i] = [255 - (i-64)*4, 255, 0]  # Decreasing blue component
            elif i < 192:  # Bright green to yellow
                heat_map[i] = [0, 255, (i-128)*4]  # Increasing red component
            else:  # Yellow to white
                heat_map[i] = [(i-192)*4, 255, 255]  # Increasing blue component
                
        # Apply custom colormap
        colorized_density = cv2.applyColorMap((norm_density * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        
        # Boost contrast further by applying a brightness adjustment
        colorized_density = cv2.convertScaleAbs(colorized_density, alpha=1.5, beta=30)
        
        # Transform the colorized density map to top view
        warped_density = cv2.warpPerspective(colorized_density, homography, size)
        
        # Apply threshold to density map
        # Create mask where density is above threshold
        mask = cv2.warpPerspective((norm_density > 0.05).astype(np.uint8) * 255, homography, size)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to warped density
        warped_density = cv2.bitwise_and(warped_density, mask)
        
        # Blend the warped density map with top view
        alpha = 0.4  # Slightly less intense than original for better point visibility
        top_view = cv2.addWeighted(top_view, 1-alpha, warped_density, alpha, 0)
        
        # Generate points based on density map
        warped_norm_density = cv2.warpPerspective(norm_density, homography, size)
        
        # Save warped density for anomaly detection
        warped_density_grayscale = warped_norm_density
        
        # Only consider points above the threshold
        high_density_areas = warped_norm_density > density_threshold
        if np.any(high_density_areas):
            # Get coordinates of high-density points
            y_coords, x_coords = np.where(high_density_areas)
            
            # Get values at these coordinates
            densities = warped_norm_density[high_density_areas]
            
            # Create list of (x, y, density) tuples
            point_data = list(zip(x_coords, y_coords, densities))
            
            # Sort by density value (highest first)
            point_data.sort(key=lambda x: x[2], reverse=True)
            
            # Cap to maximum number of points
            point_data = point_data[:max_density_points]
            
            # Store the points for later
            density_points = [(x, y) for x, y, _ in point_data]
            
            # If saving data, prepare density points data
            if save_data and frame_number is not None and area_manager is not None:
                for x, y, density_value in point_data:
                    density_points_data.append({
                        "top_view": [int(x), int(y)],
                        "density_value": float(density_value),
                        "type": "density_point"
                    })
    
    # Transform tracked people to top view and represent them as dots with IDs
    # First, get the current active IDs (people currently detected in the frame)
    current_ids = []
    object_data = []  # For saving to JSON
    detections, _ = person_detector.detect(frame)
    for detection in detections:
        if len(detection) >= 6 and detection[5] is not None:
            current_ids.append(detection[5])
    
    # Store movement vectors for visualization
    movement_vectors = []
    # Dictionary to store positions of people for anomaly visualization
    person_positions = {}
    
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
                
                # Calculate movement vector for this track (if it has at least 5 points for stability)
                if track_id in current_ids and len(top_view_track) >= 5:
                    # Use the last 5 points to calculate the average direction
                    last_points = top_view_track[-5:]
                    if len(last_points) >= 2:
                        # Calculate direction vector
                        start_point = last_points[0]
                        end_point = last_points[-1]
                        if start_point is not None and end_point is not None:
                            vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
                            vector_length = np.sqrt(vector[0]**2 + vector[1]**2)
                        else:
                            continue
                        
                        # Only include if the movement is significant
                        if vector_length > 10:  # Minimum movement threshold
                            # Normalize vector to a fixed length for visualization
                            scale = 20.0 / max(vector_length, 1e-5)  # Prevents division by zero
                            norm_vector = (int(vector[0] * scale), int(vector[1] * scale))
                            
                            # Save the vector and its starting position (current position)
                            end_point = top_view_track[-1]
                            
                            # track_id is already defined in this scope, so we use it directly
                            # No need to assign it from idx which doesn't exist here
                            
                            # Add to movement vectors with ID for anomaly detection
                            movement_vectors.append((end_point, norm_vector, color, track_id))
                            
                            # Store the position for anomaly visualization
                            person_positions[track_id] = end_point
                
                # Only draw dots for currently detected people - NO IDs
                if track_id in current_ids and len(top_view_track) > 0:
                    current_pos = top_view_track[-1]
                    
                    # Store this position for anomaly visualization even if no movement vector
                    person_positions[track_id] = current_pos
                    
                    # Draw a smaller B&W circle for the person
                    # First draw a white circle with black outline for visibility
                    cv2.circle(top_view, current_pos, 6, (0, 0, 0), 1)  # Black outline
                    cv2.circle(top_view, current_pos, 5, (255, 255, 255), -1)  # White fill
                    
                    # If saving data, prepare object data
                    if save_data and frame_number is not None and area_manager is not None:
                        # Get the original detection coordinates
                        orig_detection = None
                        for det in detections:
                            if len(det) >= 6 and det[5] == track_id:
                                orig_detection = det
                                break
                        
                        if orig_detection:
                            x1, y1, x2, y2, confidence, _ = orig_detection
                            # Make sure vector and vector_length are defined
                            if 'vector' in locals() and 'vector_length' in locals() and vector_length > 10:
                                vector_data = [int(vector[0]), int(vector[1])]
                            else:
                                vector_data = [0, 0]
                                
                            object_data.append({
                                "id": track_id,
                                "orig_bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "top_view": [current_pos[0], current_pos[1]],
                                "confidence": float(confidence),
                                "type": "tracked_person",
                                "vector": vector_data
                            })
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
                # UPDATED: Use black and white instead of color-coded IDs
                id_text = str(track_id)
            else:
                # Use a default color scheme and sequential numbering
                # Get the index in the detections list
                idx = detections.index(detection)
                id_text = f"D{idx}"  # prefix with 'D' to indicate it's not a tracked ID
            
            # UPDATED: Draw a smaller B&W dot for the detection without IDs
            # First draw a white circle with black outline for visibility
            cv2.circle(top_view, top_view_point, 6, (0, 0, 0), 1)  # Black outline
            cv2.circle(top_view, top_view_point, 5, (255, 255, 255), -1)  # White fill
            
            # If saving data, prepare object data
            if save_data and frame_number is not None and area_manager is not None:
                object_data.append({
                    "id": track_id if track_id is not None else f"D{idx}",
                    "orig_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "top_view": [top_view_point[0], top_view_point[1]],
                    "confidence": float(confidence),
                    "type": "detected_person",
                    "vector": [0, 0]  # No vector for non-tracked objects
                })
    
    # Always show average flow with a big center arrow, regardless of tracked points
    # Calculate the center of the view
    center_x, center_y = size[0] // 2, size[1] // 2
    center = (center_x, center_y)
    
    # Create a persistent big flow arrow at the center
    arrow_length = 80  # Make it quite large and prominent
    
    # Static variables for smoothing flow direction over time
    if not hasattr(create_enhanced_top_view, "avg_flow_direction"):
        create_enhanced_top_view.avg_flow_direction = (0, -arrow_length)  # Initialize with upward direction
        create_enhanced_top_view.flow_stability_counter = 0
    
    # Current direction to calculate
    current_direction = None
    
    # If we have movement vectors, calculate direction from them
    if len(movement_vectors) > 0:
        # Calculate average movement vector across all tracks
        avg_vector_x = sum(vector[0] for _, vector, _, _ in movement_vectors) / len(movement_vectors)
        avg_vector_y = sum(vector[1] for _, vector, _, _ in movement_vectors) / len(movement_vectors)
        
        # Normalize and scale to fixed length
        vector_length = np.sqrt(avg_vector_x**2 + avg_vector_y**2)
        if vector_length > 0:
            normalized_x = avg_vector_x / vector_length * arrow_length
            normalized_y = avg_vector_y / vector_length * arrow_length
            current_direction = (normalized_x, normalized_y)
        else:
            # If no movement, maintain previous direction
            current_direction = create_enhanced_top_view.avg_flow_direction
    
    # If we don't have movement vectors but have density change data, use that
    elif previous_density_map is not None and density_map is not None:
        # Calculate density change
        norm_density_current = density_map / (np.max(density_map) + 1e-10)
        norm_density_previous = previous_density_map / (np.max(previous_density_map) + 1e-10)
        
        # Find regions with significant change
        density_diff = norm_density_current - norm_density_previous
        
        # Warp the difference map to top view
        warped_diff = cv2.warpPerspective(density_diff, homography, size)
        
        # Find significant decrease and increase regions
        decrease_mask = (warped_diff < -0.1)
        increase_mask = (warped_diff > 0.1)
        
        # Calculate direction if we have significant changes
        if np.any(decrease_mask) and np.any(increase_mask):
            # Find centers of decrease and increase regions
            y_dec, x_dec = np.where(decrease_mask)
            y_inc, x_inc = np.where(increase_mask)
            
            if len(x_dec) > 0 and len(x_inc) > 0:
                # Calculate centers
                center_dec_x, center_dec_y = np.mean(x_dec), np.mean(y_dec)
                center_inc_x, center_inc_y = np.mean(x_inc), np.mean(y_inc)
                
                # Calculate direction vector
                dir_x = center_inc_x - center_dec_x
                dir_y = center_inc_y - center_dec_y
                
                # Normalize and scale
                vector_length = np.sqrt(dir_x**2 + dir_y**2)
                if vector_length > 0:
                    normalized_x = dir_x / vector_length * arrow_length
                    normalized_y = dir_y / vector_length * arrow_length
                    current_direction = (normalized_x, normalized_y)
                else:
                    # Default if no clear direction, maintain previous
                    current_direction = create_enhanced_top_view.avg_flow_direction
            else:
                # No sufficient data, maintain previous
                current_direction = create_enhanced_top_view.avg_flow_direction
        else:
            # No significant changes, maintain previous
            current_direction = create_enhanced_top_view.avg_flow_direction
    else:
        # No data available, maintain previous
        current_direction = create_enhanced_top_view.avg_flow_direction
    
    # Apply much stronger temporal smoothing to reduce jitter in flow direction
    if current_direction is not None:
        # Determine smoothing factor based on stability
        # Increase counter if direction is close to previous, reset if significant change
        prev_x, prev_y = create_enhanced_top_view.avg_flow_direction
        curr_x, curr_y = current_direction
        
        # Calculate angle difference between current and previous direction
        dot_product = prev_x * curr_x + prev_y * curr_y
        magnitude_prev = np.sqrt(prev_x**2 + prev_y**2)
        magnitude_curr = np.sqrt(curr_x**2 + curr_y**2)
        
        if magnitude_prev > 0 and magnitude_curr > 0:
            cosine_angle = dot_product / (magnitude_prev * magnitude_curr)
            # Clamp cosine to [-1, 1] to avoid numerical errors
            cosine_angle = max(-1, min(1, cosine_angle))
            angle_diff = np.arccos(cosine_angle) * 180 / np.pi
            
            # If angle change is small, increase stability counter, otherwise reduce it
            if angle_diff < 20:  # Small angle change - more strict than before
                create_enhanced_top_view.flow_stability_counter = min(40, create_enhanced_top_view.flow_stability_counter + 1)
            else:  # Large angle change
                create_enhanced_top_view.flow_stability_counter = max(0, create_enhanced_top_view.flow_stability_counter - 1)
        
        # Adjust smoothing based on stability counter
        # More stable = more smoothing (less weight to new direction)
        stability_factor = create_enhanced_top_view.flow_stability_counter / 40
        
        # Use extremely low alpha values for very stable arrow
        # Range from 0.001 (very stable) to 0.01 (less stable) - 10-100x slower changes
        alpha = 0.001 + 0.009 * (1 - stability_factor)
        
        # Exponential smoothing with variable alpha
        smoothed_x = (1 - alpha) * prev_x + alpha * curr_x
        smoothed_y = (1 - alpha) * prev_y + alpha * curr_y
        
        # Store the smoothed direction for next frame
        create_enhanced_top_view.avg_flow_direction = (smoothed_x, smoothed_y)
    
    # Use the smoothed direction for visualization
    normalized_x, normalized_y = create_enhanced_top_view.avg_flow_direction
    
    # Calculate end point for the arrow
    end_point = (int(center[0] + normalized_x), int(center[1] + normalized_y))
    
    # Draw a very prominent arrow with clear contrast
    # First draw a thicker black outline
    cv2.arrowedLine(top_view, center, end_point, (0, 0, 0), 10, tipLength=0.3, line_type=cv2.LINE_AA)
    # Then draw a slightly thinner white arrow
    cv2.arrowedLine(top_view, center, end_point, (255, 255, 255), 6, tipLength=0.3, line_type=cv2.LINE_AA)
    # Finally add a colored center for visibility
    cv2.arrowedLine(top_view, center, end_point, (0, 255, 0), 3, tipLength=0.3, line_type=cv2.LINE_AA)
    
    # Add a prominent label
    cv2.putText(top_view, "FLOW", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(top_view, "FLOW", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    # Identify and visualize crowd clusters based on density changes
    if previous_density_map is not None and density_map is not None:
        # Normalize density maps
        norm_density_current = density_map / (np.max(density_map) + 1e-10)
        norm_density_previous = previous_density_map / (np.max(previous_density_map) + 1e-10)
        
        # Create binary masks for high density areas
        current_high_density = norm_density_current > 0.4
        previous_high_density = norm_density_previous > 0.4
        
        # Warp to top view
        warped_current = cv2.warpPerspective(current_high_density.astype(np.uint8), homography, size)
        warped_previous = cv2.warpPerspective(previous_high_density.astype(np.uint8), homography, size)
        
        # Find contours in current density to identify clusters
        contours, _ = cv2.findContours(warped_current.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to find significant clusters
        min_area = 100  # Minimum area to consider as a cluster
        clusters_data = []  # For saving to JSON
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                # Draw cluster outline
                cv2.drawContours(top_view, [contour], -1, (0, 0, 0), 2)
                
                # Get the centroid of the cluster
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Label the cluster
                    cluster_id = f"C{i+1}"
                    cv2.putText(top_view, cluster_id, (cx, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(top_view, cluster_id, (cx, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Find matching cluster in previous frame
                    # This is a simplified approach - more sophisticated tracking could be used
                    prev_mask = np.zeros_like(warped_previous)
                    cv2.drawContours(prev_mask, [contour], -1, 1, -1)  # -1 means fill
                    
                    overlap = cv2.bitwise_and(prev_mask, warped_previous)
                    
                    cluster_vector = [0, 0]  # Default - no movement
                    
                    if np.any(overlap):
                        # Found a match, calculate the movement
                        prev_moments = cv2.moments(overlap)
                        if prev_moments['m00'] != 0:
                            prev_cx = int(prev_moments['m10'] / prev_moments['m00'])
                            prev_cy = int(prev_moments['m01'] / prev_moments['m00'])
                            
                            # Draw movement vector for this cluster
                            if abs(cx - prev_cx) > 3 or abs(cy - prev_cy) > 3:  # Only if significant movement
                                # Draw arrow showing cluster movement
                                cv2.arrowedLine(top_view, (prev_cx, prev_cy), (cx, cy), 
                                              (0, 0, 0), 3, tipLength=0.3)
                                cv2.arrowedLine(top_view, (prev_cx, prev_cy), (cx, cy), 
                                              (0, 255, 255), 2, tipLength=0.3)
                                
                                # Record movement vector
                                cluster_vector = [cx - prev_cx, cy - prev_cy]
                    
                    # If saving data, add this cluster
                    if save_data and frame_number is not None and area_manager is not None:
                        cluster_points = []
                        for point in contour:
                            x, y = point[0]
                            cluster_points.append([int(x), int(y)])
                            
                        clusters_data.append({
                            "id": cluster_id,
                            "centroid": [cx, cy],
                            "area": float(area),
                            "contour": cluster_points,
                            "vector": cluster_vector,
                            "type": "crowd_cluster"
                        })
    
    # Add density flow visualization if we have previous density map
    if show_flow and previous_density_map is not None and density_map is not None:
        # Calculate density change
        norm_density_current = density_map / (np.max(density_map) + 1e-10)
        norm_density_previous = previous_density_map / (np.max(previous_density_map) + 1e-10)
        
        # Find regions with significant change
        density_diff = norm_density_current - norm_density_previous
        
        # Warp the difference map to top view
        warped_diff = cv2.warpPerspective(density_diff, homography, size)
        
        # Find significant decrease and increase regions
        decrease_mask = (warped_diff < -0.1)
        increase_mask = (warped_diff > 0.1)
        
        # Only process if we have significant changes
        if np.any(decrease_mask) and np.any(increase_mask):
            # Find centers of decrease and increase regions
            y_dec, x_dec = np.where(decrease_mask)
            y_inc, x_inc = np.where(increase_mask)
            
            if len(x_dec) > 0 and len(x_inc) > 0:
                # Calculate centers
                center_dec = (int(np.mean(x_dec)), int(np.mean(y_dec)))
                center_inc = (int(np.mean(x_inc)), int(np.mean(y_inc)))
                
                # Draw a flow arrow from decrease center to increase center
                cv2.arrowedLine(top_view, center_dec, center_inc, (255, 255, 255), 4, tipLength=0.3)
                cv2.arrowedLine(top_view, center_dec, center_inc, (0, 255, 255), 2, tipLength=0.3)
                
                # Label
                cv2.putText(top_view, "Crowd Flow", 
                           ((center_dec[0] + center_inc[0])//2 + 5, 
                            (center_dec[1] + center_inc[1])//2 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Save flow data
                if save_data and frame_number is not None and area_manager is not None:
                    flow_data = {
                        "source": [center_dec[0], center_dec[1]],
                        "target": [center_inc[0], center_inc[1]],
                        "vector": [center_inc[0] - center_dec[0], center_inc[1] - center_dec[1]],
                        "type": "density_flow"
                    }
                    
                    # Append to density_points_data as it will be saved with the same function
                    density_points_data.append(flow_data)
    
    # ENHANCEMENT: Draw density-based points in the EXACT SAME style as YOLO points
    # Using white circles with black outlines to match YOLO detection points
    for i, point in enumerate(density_points):
        # Draw exactly the same size and style as YOLO points (white with black outline)
        cv2.circle(top_view, point, 6, (0, 0, 0), 1)  # Black outline
        cv2.circle(top_view, point, 5, (255, 255, 255), -1)  # White fill
    
    # Add legend for clusters
    cv2.rectangle(top_view, (size[0]-240, 10), (size[0]-10, 205), (255, 255, 255), -1)
    cv2.rectangle(top_view, (size[0]-240, 10), (size[0]-10, 205), (0, 0, 0), 1)
    cv2.putText(top_view, "Legend:", (size[0]-230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Walking area
    cv2.rectangle(top_view, (size[0]-220, 50), (size[0]-200, 60), (0, 255, 0), -1)
    cv2.putText(top_view, "Walking Area", (size[0]-190, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Road
    cv2.rectangle(top_view, (size[0]-220, 70), (size[0]-200, 80), (0, 0, 255), -1)
    cv2.putText(top_view, "Road", (size[0]-190, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # UPDATED: Active person - black and white
    cv2.circle(top_view, (size[0]-210, 100), 6, (0, 0, 0), 1)  # Black outline
    cv2.circle(top_view, (size[0]-210, 100), 5, (255, 255, 255), -1)  # White fill
    cv2.putText(top_view, "Active Person", (size[0]-190, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Historical track
    cv2.line(top_view, (size[0]-220, 120), (size[0]-200, 120), (200, 200, 200), 2)
    cv2.putText(top_view, "Historical Track", (size[0]-190, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # CHANGED: Density point (white circle with black outline)
    x, y = size[0]-210, 140
    # First draw the outline
    cv2.circle(top_view, (x, y), 8, (0, 0, 0), 2)  # Black outline
    # Then fill with white
    cv2.circle(top_view, (x, y), 6, (255, 255, 255), -1)  # White fill
    cv2.putText(top_view, "Density Point", (size[0]-190, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Heat map legend - updated color description
    cv2.rectangle(top_view, (size[0]-220, 160), (size[0]-200, 170), (0, 255, 255), -1) # Cyan color for heat map
    cv2.putText(top_view, "Density Heat Map", (size[0]-190, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add Individual Flow vector legend
    cv2.arrowedLine(top_view, (size[0]-220, 180), (size[0]-190, 180), (0, 0, 0), 3, tipLength=0.3)
    cv2.arrowedLine(top_view, (size[0]-220, 180), (size[0]-190, 180), (255, 255, 255), 2, tipLength=0.3)
    cv2.putText(top_view, "Individual Flow", (size[0]-190, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add Average Flow vector legend
    cv2.arrowedLine(top_view, (size[0]-220, 200), (size[0]-190, 200), (0, 255, 255), 3, tipLength=0.3)
    cv2.putText(top_view, "Average Flow", (size[0]-190, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Calculate average velocity for display
    avg_velocity = 0
    if len(movement_vectors) > 0:
        # Calculate average velocity magnitude from all movement vectors
        velocities = []
        for _, vector, _, _ in movement_vectors:
            vector_magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
            velocities.append(vector_magnitude)
        
        if velocities:
            avg_velocity = sum(velocities) / len(velocities)
    
    # Run anomaly detection if we have an anomaly detector and movement vectors
    anomaly_ids = set()
    if anomaly_detector is not None:
        # Prepare movement vectors for anomaly detection
        # Format: (position, vector, color, track_id)
        if len(movement_vectors) > 1:
            # Update population density heatmap in the anomaly detector using both 
            # the density map and tracked person positions
            anomaly_detector.update_population_heatmap(
                density_map=warped_density_grayscale if 'warped_density_grayscale' in locals() else None,
                person_positions=person_positions if person_positions else None
            )
            
            # Run anomaly detection
            anomaly_ids, major_flow_vector = anomaly_detector.detect_counter_flow(
                movement_vectors, 
                include_track_ids=True,
                frame_number=frame_number
            )
        else:
            # If not enough movement vectors, still get existing anomalies but don't calculate new ones
            anomaly_ids = anomaly_detector.anomalies
            major_flow_vector = (0, 0)
            # Still update the frame number for proper event tracking
            if frame_number is not None:
                anomaly_detector.current_frame = frame_number
        
        # Let the anomaly detector draw persistent anomalies on the top view
        # This ensures consistent visualization with persistence
        anomaly_detector.draw_anomaly_markers(top_view, person_positions)
        
        # Draw the major flow vector for reference
        if major_flow_vector != (0, 0):
            # Move to center of screen instead of corner
            flow_center = (size[0] // 2, size[1] // 2)
            flow_length = 30
            flow_dir_x, flow_dir_y = major_flow_vector
            
            # Normalize to desired length
            flow_magnitude = np.sqrt(flow_dir_x**2 + flow_dir_y**2)
            if flow_magnitude > 0:
                flow_dir_x = flow_dir_x / flow_magnitude * flow_length
                flow_dir_y = flow_dir_y / flow_magnitude * flow_length
                
                flow_end = (int(flow_center[0] + flow_dir_x), int(flow_center[1] + flow_dir_y))
                
                # Draw major flow arrow - make it similar to the other flow visualizations
                # First draw a thicker black outline
                cv2.arrowedLine(top_view, flow_center, flow_end, (0, 0, 0), 4, tipLength=0.3, line_type=cv2.LINE_AA)
                # Then draw a slightly thinner colored arrow
                cv2.arrowedLine(top_view, flow_center, flow_end, (0, 255, 0), 2, tipLength=0.3, line_type=cv2.LINE_AA)
                
                # Use matching text color for the major flow label
                cv2.putText(top_view, "Major Flow", (flow_center[0] - 60, flow_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        # NEW: Draw bottleneck box if identified (when anomaly count exceeds threshold)
        if anomaly_detector.bottlenecks:
            anomaly_detector.draw_bottleneck(top_view, is_original_view=False)
    
    # Add anomaly to legend
    if anomaly_detector is not None:
        # Create space for the anomaly in legend - extend existing legend area rather than creating a new one
        # Use the existing legend but make it taller to accommodate the anomaly entry
        cv2.rectangle(top_view, (size[0]-240, 10), (size[0]-10, 265), (255, 255, 255), -1)
        cv2.rectangle(top_view, (size[0]-240, 10), (size[0]-10, 265), (0, 0, 0), 1)
        cv2.putText(top_view, "Legend:", (size[0]-230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Walking area
        cv2.rectangle(top_view, (size[0]-220, 50), (size[0]-200, 60), (0, 255, 0), -1)
        cv2.putText(top_view, "Walking Area", (size[0]-190, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Road
        cv2.rectangle(top_view, (size[0]-220, 70), (size[0]-200, 80), (0, 0, 255), -1)
        cv2.putText(top_view, "Road", (size[0]-190, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Active person
        cv2.circle(top_view, (size[0]-210, 100), 6, (0, 0, 0), 1)  # Black outline
        cv2.circle(top_view, (size[0]-210, 100), 5, (255, 255, 255), -1)  # White fill
        cv2.putText(top_view, "Active Person", (size[0]-190, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Historical track
        cv2.line(top_view, (size[0]-220, 120), (size[0]-200, 120), (200, 200, 200), 2)
        cv2.putText(top_view, "Historical Track", (size[0]-190, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Density point
        x, y = size[0]-210, 140
        cv2.circle(top_view, (x, y), 8, (0, 0, 0), 2)  # Black outline
        cv2.circle(top_view, (x, y), 6, (255, 255, 255), -1)  # White fill
        cv2.putText(top_view, "Density Point", (size[0]-190, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Heat map legend
        cv2.rectangle(top_view, (size[0]-220, 160), (size[0]-200, 170), (0, 255, 255), -1) # Cyan color for heat map
        cv2.putText(top_view, "Density Heat Map", (size[0]-190, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Individual Flow vector legend
        cv2.arrowedLine(top_view, (size[0]-220, 180), (size[0]-190, 180), (0, 0, 0), 3, tipLength=0.3)
        cv2.arrowedLine(top_view, (size[0]-220, 180), (size[0]-190, 180), (255, 255, 255), 2, tipLength=0.3)
        cv2.putText(top_view, "Individual Flow", (size[0]-190, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Average Flow vector legend
        cv2.arrowedLine(top_view, (size[0]-220, 200), (size[0]-190, 200), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(top_view, "Average Flow", (size[0]-190, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Major Flow vector legend
        cv2.arrowedLine(top_view, (size[0]-220, 220), (size[0]-190, 220), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(top_view, "Major Flow", (size[0]-190, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add anomaly legend entry
        x, y = size[0]-210, 240
        diamond_size = 8
        diamond_points = np.array([
            [x, y - diamond_size],
            [x + diamond_size, y],
            [x, y + diamond_size],
            [x - diamond_size, y]
        ], np.int32)
        
        # Draw blue diamond for anomaly
        cv2.fillPoly(top_view, [diamond_points], (255, 0, 0))  # Changed to blue (BGR)
        cv2.polylines(top_view, [diamond_points], True, (255, 0, 0), 2)  # Changed to blue (BGR)
        cv2.putText(top_view, "Anomaly", (size[0]-190, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add anomaly count
        cv2.putText(top_view, f"Anomalies: {len(anomaly_ids)}", (10, size[1] - 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Changed to blue (BGR)
    
    # Add current count
    num_active_people = len([d for d in detections if len(d) >= 6 and d[5] is not None])
    cv2.putText(top_view, f"Active People: {num_active_people}", (10, size[1] - 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add average velocity to left-side labels
    cv2.putText(top_view, f"Avg Velocity: {avg_velocity:.1f}", (10, size[1] - 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add density points count
    cv2.putText(top_view, f"Density Points: {len(density_points)}", (10, size[1] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add density count information
    cv2.putText(top_view, f"Estimated people: {estimated_count*0.05:.1f}", (10, size[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save data if requested
    if save_data and frame_number is not None and area_manager is not None:
        # Save detected objects
        if object_data:
            area_manager.save_detected_objects(frame_number, object_data)
        
        # Save density points
        if density_points_data:
            area_manager.save_density_points(frame_number, density_points_data)
        
        # Save clusters data as special density points
        if 'clusters_data' in locals() and clusters_data:
            # Append clusters to density points data
            for cluster in clusters_data:
                density_points_data.append(cluster)
            
            # Save updated density points data
            area_manager.save_density_points(frame_number, density_points_data)
    
    return top_view


# New function to visualize crowd movement over time
def create_flow_visualization(frame, density_map_current, density_map_previous, person_detector, 
                             homography, area_manager, size=(800, 600)):
    """
    Create a visualization showing the movement of crowd density over time and the flow vectors
    
    Parameters:
    - frame: Input frame
    - density_map_current: Current frame density map
    - density_map_previous: Previous frame density map
    - person_detector: Instance of PersonDetector
    - homography: Perspective transformation matrix
    - area_manager: Instance of AreaManager
    - size: Size of the top view image (width, height)
    
    Returns:
    - flow_view: Visualization of crowd movement
    """
    # Use static variables to store flow history and arrow properties between calls
    if not hasattr(create_flow_visualization, "avg_flow_history"):
        create_flow_visualization.avg_flow_history = []
    if not hasattr(create_flow_visualization, "crowd_position"):
        create_flow_visualization.crowd_position = None
    if not hasattr(create_flow_visualization, "crowd_vector"):
        create_flow_visualization.crowd_vector = None
    if not hasattr(create_flow_visualization, "position_stability"):
        create_flow_visualization.position_stability = 0
    if not hasattr(create_flow_visualization, "last_update_frame"):
        create_flow_visualization.last_update_frame = 0
        
    # Set parameters
    history_length = 5       # Number of frames for flow history
    update_interval = 120    # Minimum frames between direction updates (increased for more stability)
    smoothing_factor = 0.1   # How quickly to blend in new direction (reduced for more stability)
    # Use the static flow history
    avg_flow_history = create_flow_visualization.avg_flow_history
    
    # Create a blank top-view image
    flow_view = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Draw a grid for reference
    for x in range(0, size[0], 50):
        cv2.line(flow_view, (x, 0), (x, size[1]), (200, 200, 200), 1)
    for y in range(0, size[1], 50):
        cv2.line(flow_view, (0, y), (size[0], y), (200, 200, 200), 1)
    
    # Transform walking areas and roads to top view
    top_view_walking_areas = []
    for area in area_manager.walking_areas:
        top_view_area = transform_polygon(area, homography)
        top_view_walking_areas.append(top_view_area)
        cv2.fillPoly(flow_view, [top_view_area], (0, 255, 0, 128))
        cv2.polylines(flow_view, [top_view_area], True, (0, 255, 0), 2)
    
    top_view_roads = []
    for road in area_manager.roads:
        top_view_road = transform_polygon(road, homography)
        top_view_roads.append(top_view_road)
        cv2.fillPoly(flow_view, [top_view_road], (0, 0, 255, 128))
        cv2.polylines(flow_view, [top_view_road], True, (0, 0, 255), 2)
    
    # Variable to store crowd movement vector for this frame
    crowd_movement_vector = None
    crowd_movement_center = None
    
    # Calculate density flow between previous and current frame
    if density_map_current is not None and density_map_previous is not None:
        # Normalize density maps
        norm_density_current = density_map_current / (np.max(density_map_current) + 1e-10)
        norm_density_previous = density_map_previous / (np.max(density_map_previous) + 1e-10)
        
        # Calculate difference
        density_diff = norm_density_current - norm_density_previous
        
        # Colorize the difference map (blue for decrease, red for increase)
        diff_colormap = np.zeros((density_diff.shape[0], density_diff.shape[1], 3), dtype=np.uint8)
        
        # Blue for decreasing density, red for increasing
        diff_colormap[density_diff < -0.05] = [255, 0, 0]   # Blue for decreasing
        diff_colormap[density_diff > 0.05] = [0, 0, 255]    # Red for increasing
        
        # Transform the difference map to top view
        warped_diff = cv2.warpPerspective(diff_colormap, homography, size)
        
        # Apply the warped difference map to the flow view
        alpha = 0.5
        mask = np.any(warped_diff != [0, 0, 0], axis=2).astype(np.uint8) * 255
        mask = np.stack([mask, mask, mask], axis=2)
        masked_diff = cv2.bitwise_and(warped_diff, mask)
        flow_view = cv2.addWeighted(flow_view, 1.0, masked_diff, alpha, 0)
        
        # Calculate flow vectors based on density changes
        # Downscale density maps for flow calculation
        scale_factor = 8  # Process 8x smaller maps for efficiency
        small_current = cv2.resize(norm_density_current, 
                                 (norm_density_current.shape[1]//scale_factor, 
                                  norm_density_current.shape[0]//scale_factor))
        small_previous = cv2.resize(norm_density_previous, 
                                   (norm_density_previous.shape[1]//scale_factor, 
                                    norm_density_previous.shape[0]//scale_factor))
        
        # Only calculate flow for areas with significant density
        threshold = 0.1
        mask_current = small_current > threshold
        mask_previous = small_previous > threshold
        
        # Find centers of mass for density regions
        if np.any(mask_current) and np.any(mask_previous):
            # Get points above threshold
            y_curr, x_curr = np.where(mask_current)
            y_prev, x_prev = np.where(mask_previous)
            
            # Calculate center of mass for each frame
            if len(x_curr) > 0 and len(x_prev) > 0:
                center_curr_x = np.mean(x_curr) * scale_factor
                center_curr_y = np.mean(y_curr) * scale_factor
                center_prev_x = np.mean(x_prev) * scale_factor
                center_prev_y = np.mean(y_prev) * scale_factor
                
                # Calculate displacement vector
                dx = center_curr_x - center_prev_x
                dy = center_curr_y - center_prev_y
                
                # Transform centers to top view
                top_curr = transform_point((center_curr_x, center_curr_y), homography)
                top_prev = transform_point((center_prev_x, center_prev_y), homography)
                
                # Store crowd movement vector for this frame
                crowd_movement_center = top_curr
                crowd_movement_vector = (top_curr[0] - top_prev[0], top_curr[1] - top_prev[1])
    
    # Get vectors from individual tracked people
    movement_vectors = []
    
    if person_detector.tracker_enabled:
        for track_id, track in person_detector.track_history.items():
            if len(track) > 4:  # Need at least 5 points for stable movement
                # Transform last 5 track points to top view
                top_view_track = []
                for point in track[-5:]:
                    top_view_point = transform_point(point, homography)
                    top_view_track.append(top_view_point)
                
                # Calculate movement vector
                if len(top_view_track) >= 2:
                    start_point = top_view_track[0]
                    end_point = top_view_track[-1]
                    vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
                    
                    # Only include significant movement
                    vector_length = np.sqrt(vector[0]**2 + vector[1]**2)
                    if vector_length > 10:
                        # Use a fixed length for visualization
                        scale = 30.0 / max(vector_length, 1e-5)
                        norm_vector = (int(vector[0] * scale), int(vector[1] * scale))
                        
                        # Save the vector with color
                        color = person_detector._get_color_by_id(track_id)
                        movement_vectors.append((end_point, norm_vector, color, track_id))
    
    # Draw individual movement vectors
    for pos, vector, color, _ in movement_vectors:
        end_point = (pos[0] + vector[0], pos[1] + vector[1])
        cv2.arrowedLine(flow_view, pos, end_point, color, 2, tipLength=0.3)
    
    # Calculate current frame's average flow vector if we have enough data
    current_avg_flow = None
    if len(movement_vectors) >= 3:
        avg_x = sum(v[0] for _, v, _, _ in movement_vectors) / len(movement_vectors)
        avg_y = sum(v[1] for _, v, _, _ in movement_vectors) / len(movement_vectors)
        
        # Calculate center position
        center_x = sum(p[0] for p, _, _, _ in movement_vectors) / len(movement_vectors)
        center_y = sum(p[1] for p, _, _, _ in movement_vectors) / len(movement_vectors)
        center = (int(center_x), int(center_y))
        
        current_avg_flow = {
            'center': center,
            'vector': (avg_x, avg_y)
        }
        
        # Add to history
        avg_flow_history.append(current_avg_flow)
        
        # Keep history at specified length
        while len(avg_flow_history) > history_length:
            avg_flow_history.pop(0)
    
    # Draw persistent average flow vector using history
    if avg_flow_history:
        # Calculate average of historical vectors
        avg_x = sum(flow['vector'][0] for flow in avg_flow_history) / len(avg_flow_history)
        avg_y = sum(flow['vector'][1] for flow in avg_flow_history) / len(avg_flow_history)
        
        # Calculate average center
        center_x = sum(flow['center'][0] for flow in avg_flow_history) / len(avg_flow_history)
        center_y = sum(flow['center'][1] for flow in avg_flow_history) / len(avg_flow_history)
        center = (int(center_x), int(center_y))
        
        # Draw average vector
        end = (int(center[0] + avg_x*1.5), int(center[1] + avg_y*1.5))
        cv2.arrowedLine(flow_view, center, end, (0, 255, 255), 4, tipLength=0.3)
        
        # Add text with offset to avoid overlap
        text_pos = (end[0] + 10, end[1] - 10)
        cv2.putText(flow_view, "Average Flow", text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw crowd movement vector if available
    if crowd_movement_vector and crowd_movement_center:
        # Get current frame number (using a basic counter)
        if not hasattr(create_flow_visualization, "frame_count"):
            create_flow_visualization.frame_count = 0
        create_flow_visualization.frame_count += 1
        current_frame = create_flow_visualization.frame_count
        
        # Check if this is our first vector or if we've waited long enough to update
        first_vector = create_flow_visualization.crowd_vector is None
        enough_frames_passed = (current_frame - create_flow_visualization.last_update_frame) >= update_interval
        
        if first_vector:
            # First time - just store the vector
            create_flow_visualization.crowd_position = crowd_movement_center
            create_flow_visualization.crowd_vector = crowd_movement_vector
            create_flow_visualization.last_update_frame = current_frame
        elif enough_frames_passed:
            # Time to update - smoothly transition to new direction
            old_x, old_y = create_flow_visualization.crowd_vector
            new_x, new_y = crowd_movement_vector
            
            # Apply exponential smoothing
            smoothed_x = old_x * (1 - smoothing_factor) + new_x * smoothing_factor
            smoothed_y = old_y * (1 - smoothing_factor) + new_y * smoothing_factor
            
            # Update the stored vector
            create_flow_visualization.crowd_vector = (smoothed_x, smoothed_y)
            create_flow_visualization.crowd_position = crowd_movement_center
            create_flow_visualization.last_update_frame = current_frame
        
    # Draw the crowd movement arrow using our stored stable vector
    if create_flow_visualization.crowd_position is not None and create_flow_visualization.crowd_vector is not None:
        display_center = create_flow_visualization.crowd_position
        display_vector = create_flow_visualization.crowd_vector
        
        # Scale vector for visibility
        vector_length = np.sqrt(display_vector[0]**2 + display_vector[1]**2)
        if vector_length > 5:  # Only draw significant movement
            scale = 40.0 / max(vector_length, 1e-5)
            scaled_vector = (
                int(display_vector[0] * scale),
                int(display_vector[1] * scale)
            )
            
            end_point = (
                display_center[0] + scaled_vector[0],
                display_center[1] + scaled_vector[1]
            )
            
            cv2.arrowedLine(flow_view, display_center, end_point, (255, 0, 255), 3, tipLength=0.3)
            
            # Add text with offset to avoid overlap
            text_pos = (end_point[0] + 10, end_point[1] + 10)
            cv2.putText(flow_view, "Crowd Movement", text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Add legend with improved layout
    legend_width = 250
    legend_height = 170
    legend_x = size[0] - legend_width - 10
    legend_y = 10
    
    # Create semi-transparent background for legend
    overlay = flow_view.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), 
                 (legend_x + legend_width, legend_y + legend_height), 
                 (240, 240, 240), -1)
    
    # Apply transparency
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, flow_view, 1 - alpha, 0, flow_view)
    
    # Add border
    cv2.rectangle(flow_view, (legend_x, legend_y), 
                 (legend_x + legend_width, legend_y + legend_height), 
                 (0, 0, 0), 1)
    
    # Add legend title
    cv2.putText(flow_view, "Flow Legend:", 
               (legend_x + 10, legend_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Calculate vertical spacing
    line_height = 30
    start_y = legend_y + 50
    
    # Individual movement
    cv2.arrowedLine(flow_view, 
                   (legend_x + 20, start_y), 
                   (legend_x + 50, start_y), 
                   (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(flow_view, "Individual Movement", 
               (legend_x + 60, start_y + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Average flow
    cv2.arrowedLine(flow_view, 
                   (legend_x + 20, start_y + line_height), 
                   (legend_x + 50, start_y + line_height), 
                   (0, 255, 255), 3, tipLength=0.3)
    cv2.putText(flow_view, "Average Flow", 
               (legend_x + 60, start_y + line_height + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Crowd movement
    cv2.arrowedLine(flow_view, 
                   (legend_x + 20, start_y + 2*line_height), 
                   (legend_x + 50, start_y + 2*line_height), 
                   (255, 0, 255), 3, tipLength=0.3)
    cv2.putText(flow_view, "Crowd Movement", 
               (legend_x + 60, start_y + 2*line_height + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Density change
    cv2.rectangle(flow_view, 
                 (legend_x + 20, start_y + 3*line_height - 5), 
                 (legend_x + 40, start_y + 3*line_height + 5), 
                 (255, 0, 0), -1)
    cv2.putText(flow_view, "Density Decrease", 
               (legend_x + 60, start_y + 3*line_height + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(flow_view, 
                 (legend_x + 20, start_y + 4*line_height - 5), 
                 (legend_x + 40, start_y + 4*line_height + 5), 
                 (0, 0, 255), -1)
    cv2.putText(flow_view, "Density Increase", 
               (legend_x + 60, start_y + 4*line_height + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Update static variable for next call
    create_flow_visualization.avg_flow_history = avg_flow_history
    
    return flow_view
def enhanced_process_cctv_to_top_view(video_path, output_path=None, calibration_image=None, 
                                      src_points=None, use_tracking=True, yolo_model_size='x', 
                                      csrnet_model_path=None, density_threshold=0.2, max_points=200,
                                      save_data=True, load_saved_data=True, preprocess_video=True,
                                      anomaly_threshold=30, stampede_threshold=35, max_bottlenecks=3):
    """
    Process CCTV footage to create an enhanced top-view simulation with advanced visualizations
    
    Parameters:
    - video_path: Path to input CCTV video
    - output_path: Path for output video (if None, don't save)
    - calibration_image: Path to image for calibration (if None, use first frame)
    - src_points: Four corner points in the source image (if None, prompt user or load from saved data)
    - use_tracking: Whether to enable person tracking
    - yolo_model_size: Size of YOLO model ('n', 's', 'm', 'l', 'x')
    - csrnet_model_path: Path to CSRNet pre-trained weights (if None, use default)
    - density_threshold: Threshold for density points
    - max_points: Maximum number of density points
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
    
    # Initialize anomaly detector for counter-flow detection with persistence
    print("Setting up Anomaly Detector...")
    # Use angle_threshold=65 degrees, history_length=5 frames, anomaly_persistence=60 frames
    # and anomaly_threshold for bottleneck detection (default: 30)
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
        out_top_view = cv2.VideoWriter(f"{base_path}_{source_video_name}_enhanced_top_view.mp4", fourcc, fps, top_view_size)
        out_original = cv2.VideoWriter(f"{base_path}_{source_video_name}_enhanced_original.mp4", fourcc, fps, (width, height))
        
        # Only create density output if we have a density estimator
        if crowd_estimator is not None:
            out_density = cv2.VideoWriter(f"{base_path}_{source_video_name}_enhanced_density.mp4", fourcc, fps, (width, height))
        
        print(f"Output videos will be saved with prefix: {source_video_name} (enhanced)")

    # Create directories for saving snapshots
    snapshots_dir = None
    if output_path:
        # Use the source video name in the snapshots directory name
        source_video_name = os.path.splitext(os.path.basename(video_path))[0]
        base_path = os.path.splitext(output_path)[0]
        snapshots_dir = f"{base_path}_{source_video_name}_enhanced_snapshots"
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
    print("Enhanced Mode Controls:")
    print("  'q' - Quit the application")
    print("  'r' - Reset heat maps")
    print("  't' - Toggle tracking visualization (on/off)")
    print("  '+' - Increase density threshold (show fewer points)")
    print("  '-' - Decrease density threshold (show more points)")
    print("  'p' - Toggle display of density points")
    print("  'f' - Toggle flow visualization (on/off)")
    print("  's' - Toggle data saving (on/off)")
    print("\nPoints in the top view visualize both detected people and estimated crowd density.")
    print("Yellow stars represent density-based points from CSRNet estimation.")
    print("Flow visualization shows movement direction of crowds and individuals.")
    print("Anomaly detection highlights entities moving against the main flow with blue diamonds.")
    print("Anomalies persist for 60 frames (~2 seconds at 30fps) for better visibility.")
    print("Anomaly detection information (ID, time, frame) is stored in 'video_data/anomalies' as JSON files.")
    print("\nNote: When data saving is ON, data is saved every 20 frames to reduce disk usage.")
    
    # For FPS calculation
    start_time = time.time()
    frame_time = start_time
    
    # Create windows with fixed positions for better visualization
    cv2.namedWindow('Enhanced Original with Density and Detections', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Enhanced Original with Density and Detections', 50, 50)
    cv2.resizeWindow('Enhanced Original with Density and Detections', 640, 480)
    
    cv2.namedWindow('Enhanced Top View with Anomaly Detection', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Enhanced Top View with Anomaly Detection', 700, 50)
    cv2.resizeWindow('Enhanced Top View with Anomaly Detection', 800, 600)
    
    if crowd_estimator is not None:
        cv2.namedWindow('Enhanced Crowd Density Heat Map', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Enhanced Crowd Density Heat Map', 50, 550)
        cv2.resizeWindow('Enhanced Crowd Density Heat Map', 640, 480)
    
    # Additional state flags for enhanced mode
    show_density_points = True  # Toggle for showing density points
    show_flow = True  # Toggle for showing flow visualization
    current_density_threshold = density_threshold
    current_max_points = max_points
    current_save_data = save_data  # Toggle for data saving
    
    # Store previous density map for flow calculation
    previous_density_map = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - frame_time
        frame_time = current_time
        fps_current = 1.0 / elapsed if elapsed > 0 else 0
        
        # Check if we have saved data for this frame if load_saved_data is enabled
        saved_objects = None
        saved_density_points = None
        
        if load_saved_data:
            saved_objects = area_manager.load_detected_objects(frame_count)
            if crowd_estimator is not None:
                saved_density_points = area_manager.load_density_points(frame_count)
        
        # First, estimate crowd density (if available)
        density_map = None
        estimated_count = 0
        colorized_density = None
        
        if crowd_estimator is not None:
            try:
                density_map, estimated_count, colorized_density = crowd_estimator.estimate_density(frame)
            except Exception as e:
                print(f"Error in density estimation: {e}")
                # Continue without density estimation for this frame
        
        # Then, detect and track individual people with YOLOv8
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
        
        # 3. Create enhanced top view with density map, tracking, flow, and density points
        # Only save data every 20 frames to reduce disk usage
        save_current_frame = current_save_data and (frame_count % 20 == 0)
        
        top_view = create_enhanced_top_view(
            frame, 
            density_map if show_density_points else None,
            previous_density_map,  # Pass the previous density map for flow calculation
            person_detector, 
            homography, 
            area_manager, 
            estimated_count, 
            top_view_size,
            current_density_threshold,
            current_max_points,
            show_flow,  # Pass the show_flow flag
            frame_count if save_current_frame else None,
            save_current_frame,
            anomaly_detector  # Pass the anomaly detector
        )
        
        # Store current density map for next frame
        if density_map is not None:
            previous_density_map = density_map.copy()
        
        # Add detection information to original view
        # Now includes the number of people with IDs
        person_count_with_ids = sum(1 for det in detections if len(det) >= 6 and det[5] is not None)
        cv2.putText(frame_with_visualization, f"YOLO detections: {len(detections)}", (10, height - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_with_visualization, f"People with IDs: {person_count_with_ids}", (10, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_with_visualization, f"Density estimate: {estimated_count*0.05:.1f}", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add enhanced mode indicators
        cv2.putText(frame_with_visualization, "ENHANCED MODE", (width - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add density threshold information
        cv2.putText(frame_with_visualization, f"Density threshold: {current_density_threshold:.2f}", (width - 250, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # NEW: Project and draw bottleneck from top view to original frame if identified
        if anomaly_detector.bottlenecks:
            # Project bottleneck locations to original frame
            original_bottlenecks = anomaly_detector.project_bottleneck_to_original(inv_homography)
            if original_bottlenecks is not None:
                # Draw bottlenecks on original frame
                anomaly_detector.draw_bottleneck(frame_with_visualization, is_original_view=True, 
                                                original_bottlenecks=original_bottlenecks)
        
        # Add data saving indicator
        save_text = "Data Saving: ON" if current_save_data else "Data Saving: OFF"
        cv2.putText(frame_with_visualization, save_text, (width - 200, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if current_save_data else (0, 0, 255), 2)
        
        # Add FPS display
        cv2.putText(frame_with_visualization, f"FPS: {fps_current:.1f}", (width - 150, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frames
        cv2.imshow('Enhanced Original with Density and Detections', frame_with_visualization)
        
        # Display separate density map if available
        if colorized_density is not None:
            cv2.imshow('Enhanced Crowd Density Heat Map', colorized_density)
        
        # Display top view
        cv2.imshow('Enhanced Top View with Anomaly Detection', top_view)
        
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
            # Save all pending anomaly data before exiting
            if anomaly_detector is not None:
                print("Saving all pending anomaly data...")
                anomaly_detector.save_all_anomaly_data()
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
        elif key == ord('p'):
            # Toggle density points on/off if 'p' is pressed
            show_density_points = not show_density_points
            print(f"Density points {'enabled' if show_density_points else 'disabled'}")
        elif key == ord('f'):
            # Toggle flow visualization on/off if 'f' is pressed
            show_flow = not show_flow
            print(f"Flow visualization {'enabled' if show_flow else 'disabled'}")
        elif key == ord('s'):
            # Toggle data saving on/off if 's' is pressed
            current_save_data = not current_save_data
            print(f"Data saving {'enabled' if current_save_data else 'disabled'}")
        elif key == ord('+') or key == ord('='):
            # Increase density threshold (fewer points)
            current_density_threshold = min(current_density_threshold + 0.05, 0.95)
            print(f"Density threshold increased to {current_density_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            # Decrease density threshold (more points)
            current_density_threshold = max(current_density_threshold - 0.05, 0.05)
            print(f"Density threshold decreased to {current_density_threshold:.2f}")
        
        # Save snapshots at regular intervals
        if snapshots_dir and frame_count % snapshot_interval == 0:
            timestamp = time.strftime("%H%M%S")
            
            try:
                # Save views
                for view_name, view_img in [
                    ("top_view", top_view),
                    ("original", frame_with_visualization),
                ]:
                    view_dir = os.path.join(snapshots_dir, view_name)
                    if not os.path.exists(view_dir):
                        os.makedirs(view_dir)
                    
                    cv2.imwrite(os.path.join(view_dir, f"frame_{frame_count:06d}_{timestamp}.jpg"), 
                              view_img)
                
                # Save density view if available
                if colorized_density is not None and crowd_estimator is not None:
                    density_dir = os.path.join(snapshots_dir, "density")
                    if not os.path.exists(density_dir):
                        os.makedirs(density_dir)
                    
                    cv2.imwrite(os.path.join(density_dir, f"frame_{frame_count:06d}_{timestamp}.jpg"), 
                              colorized_density)
                    
                    # Save raw density data
                    if density_map is not None:
                        np.save(os.path.join(density_dir, f"raw_density_{frame_count:06d}_{timestamp}.npy"), 
                              density_map)
                
            except Exception as e:
                print(f"Warning: Error saving snapshots for frame {frame_count}: {e}")
        
        frame_count += 1
    
    # Release resources
    cap.release()
    
    # Save any pending anomaly data before closing
    if anomaly_detector is not None:
        print("Saving all pending anomaly data...")
        anomaly_detector.save_all_anomaly_data()
    
    # Close all video writers
    for writer in [out_top_view, out_original, out_density]:
        if writer:
            writer.release()
    
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete in {elapsed_time:.2f} seconds.")
    
    # Report output files
    if output_path:
        if os.path.isdir(output_path):
            base_path = os.path.join(output_path, source_video_name)
        else:
            base_path = os.path.splitext(output_path)[0]
        
        print(f"Output saved to:")
        print(f"  - {base_path}_{source_video_name}_enhanced_top_view.mp4 (Top view with tracking and flow)")
        print(f"  - {base_path}_{source_video_name}_enhanced_original.mp4 (Original view with detections)")
        if crowd_estimator is not None:
            print(f"  - {base_path}_{source_video_name}_enhanced_density.mp4 (Crowd density heat map)")
        print(f"  - {snapshots_dir} (Directory with snapshots and raw data)")
    
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
    
    # Return the paths to the simulation videos
    simulation_paths = {
        "top_view": f"{base_path}_{source_video_name}_enhanced_top_view.mp4" if output_path else None,
        "original": f"{base_path}_{source_video_name}_enhanced_original.mp4" if output_path else None,
        "density": f"{base_path}_{source_video_name}_enhanced_density.mp4" if output_path and crowd_estimator is not None else None
    }
    
    return simulation_paths

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crowd Simulation Enhancement Tool')
    
    # Input source options
    parser.add_argument('--input_video', type=str, default=None, help='Path to the input video file')
    parser.add_argument('--input_folder', type=str, default=None, help='Path to folder containing image sequence')
    parser.add_argument('--enable_webcam', action='store_true', help='Use webcam as input source')
    parser.add_argument('--webcam_id', type=int, default=0, help='Webcam device ID to use')
    
    # Processing options
    parser.add_argument('--frame_skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--use_tracker', action='store_true', help='Use object tracker for smoother detection')
    parser.add_argument('--show_tracks', action='store_true', help='Show tracking lines for pedestrians')
    parser.add_argument('--track_length', type=int, default=15, help='Length of tracking lines')
    parser.add_argument('--perspective_transform', action='store_true', help='Apply perspective transform for top-down view')
    parser.add_argument('--show_density', action='store_true', help='Show crowd density heatmap')
    parser.add_argument('--perspective_file', type=str, default=None, help='Path to perspective transform points file')
    parser.add_argument('--max_people', type=int, default=None, help='Maximum number of people to detect')
    parser.add_argument('--anomaly_threshold', type=int, default=30, help='Threshold for identifying bottlenecks')
    parser.add_argument('--stampede_threshold', type=int, default=35, help='Threshold for triggering stampede warnings')
    parser.add_argument('--enable_preprocessing', action='store_true', help='Enable video preprocessing (compression/resizing)')
    
    # Output options
    parser.add_argument('--output_video', type=str, default=None, help='Path to save output video')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to save output frames')
    parser.add_argument('--display_scale', type=float, default=1.0, help='Scale factor for display windows')
    parser.add_argument('--fps_overlay', action='store_true', help='Show FPS overlay on output')
    
    return parser.parse_args()

def main():
    """Main function to run the simulation enhancement."""
    args = parse_arguments()
    
    # Input validation
    if not args.input_video and not args.input_folder and not args.enable_webcam:
        print("Error: No input source specified. Use --input_video, --input_folder, or --enable_webcam")
        sys.exit(1)
    
    # Set up logging
    setup_logging()
    
    # Initialize the detector
    detector = PedestrianDetector(
        confidence_threshold=0.4,
        use_tracker=args.use_tracker
    )
    
    # Initialize the anomaly detector with command-line threshold
    anomaly_detector = AnomalyDetector(
        anomaly_threshold=args.anomaly_threshold,
        stampede_threshold=args.stampede_threshold
    )
    
    # Initialize the density estimator
    density_estimator = DensityEstimator()

    # ... rest of the code ...