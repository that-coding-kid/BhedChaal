#!/usr/bin/env python
"""
Enhanced main script for running the CCTV to top view conversion with 
density points and detection heat map
"""

import os
import argparse
from visualization import process_cctv_to_top_view
from simulation_enhancement import enhanced_process_cctv_to_top_view

def main():
    """
    Main function for the enhanced crowd analysis application
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process CCTV footage for crowd analysis with density points')
    parser.add_argument('--video-path', '-v', type=str, default='japan.mp4',
                        help='Path to input video file (default: japan.mp4)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path for output file (default: None, only display)')
    parser.add_argument('--calibration', '-c', type=str, default=None,
                        help='Path to calibration image (default: use first frame)')
    parser.add_argument('--model-size', '-m', type=str, default='x', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (default: x)')
    parser.add_argument('--csrnet-weights', '-w', type=str, default=None,
                        help='Path to CSRNet weights (default: use built-in default)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable person tracking (default: tracking enabled)')
    parser.add_argument('--enhanced', '-e', action='store_true',
                        help='Use enhanced visualization with density points (default: False)')
    parser.add_argument('--density-threshold', type=float, default=0.2,
                        help='Threshold for density-based data points (default: 0.2)')
    parser.add_argument('--max-points', type=int, default=200,
                        help='Maximum number of density points (default: 200)')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Disable video preprocessing/standardization (default: preprocessing enabled)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return 1
    
    # Check if calibration image exists (if provided)
    if args.calibration and not os.path.exists(args.calibration):
        print(f"Error: Calibration image '{args.calibration}' not found.")
        return 1
    
    # Check if CSRNet weights exist (if provided)
    if args.csrnet_weights and not os.path.exists(args.csrnet_weights):
        print(f"Warning: CSRNet weights '{args.csrnet_weights}' not found.")
        print("Continuing with default weights.")
    
    # Create output directory if needed
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Process the video
    print(f"Processing video: {args.video_path}")
    print(f"Output path: {args.output if args.output else 'None (display only)'}")
    print(f"Tracking: {'Disabled' if args.no_tracking else 'Enabled'}")
    print(f"YOLOv8 model size: {args.model_size}")
    print(f"Using enhanced visualization: {'Yes' if args.enhanced else 'No'}")
    print(f"Video preprocessing: {'Disabled' if args.no_preprocess else 'Enabled'}")
    
    if args.enhanced:
        print(f"Density threshold: {args.density_threshold}")
        print(f"Max density points: {args.max_points}")
    
    try:
        # Run the processing with appropriate function
        if args.enhanced:
            # Use the enhanced function with density points
            result = enhanced_process_cctv_to_top_view(
                args.video_path,
                args.output,
                args.calibration,
                use_tracking=not args.no_tracking,
                yolo_model_size=args.model_size,
                csrnet_model_path=args.csrnet_weights,
                density_threshold=args.density_threshold,
                max_points=args.max_points,
                preprocess_video=not args.no_preprocess
            )
        else:
            # Use the original function
            result = process_cctv_to_top_view(
                args.video_path,
                args.output,
                args.calibration,
                use_tracking=not args.no_tracking,
                yolo_model_size=args.model_size,
                csrnet_model_path=args.csrnet_weights,
                preprocess_video=not args.no_preprocess
            )
        
        if result:
            print(f"Processing completed successfully. Output: {result}")
            return 0
        else:
            print("Processing completed but no output was produced.")
            return 0
            
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Set environment variables for OpenCV to avoid conflicts with older CUDA libraries
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Run the main function
    exit_code = main()
    exit(exit_code)