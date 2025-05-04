#!/usr/bin/env python
"""
Enhanced main script for running the CCTV to top view conversion with 
density points and detection heat map
"""

import os
import argparse
import subprocess
import sys
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
    parser.add_argument('--anomaly-threshold', type=int, default=30,
                        help='Threshold for bottleneck detection when anomalies exceed this value (default: 30)')
    parser.add_argument('--stampede-threshold', type=int, default=35,
                        help='Threshold for stampede warning when anomalies exceed this value (default: 35)')
    parser.add_argument('--max-bottlenecks', type=int, default=3,
                        help='Maximum number of bottlenecks to identify (default: 3)')
    parser.add_argument('--run-panic-sim', action='store_true',
                        help='Launch panic/stampede simulation after video processing')
    parser.add_argument('--auto-record', action='store_true',
                        help='Automatically start recording animation when panic simulation begins')
    parser.add_argument('--record-duration', type=int, default=60,
                        help='Duration of auto-recording in seconds (default: 60)')
    
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
        print(f"Anomaly threshold: {args.anomaly_threshold}")
        print(f"Stampede threshold: {args.stampede_threshold}")
        print(f"Max bottlenecks: {args.max_bottlenecks}")
    
    if args.run_panic_sim and args.auto_record:
        print(f"Auto-recording: Enabled ({args.record_duration} seconds)")
    
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
                preprocess_video=not args.no_preprocess,
                anomaly_threshold=args.anomaly_threshold,
                stampede_threshold=args.stampede_threshold,
                max_bottlenecks=args.max_bottlenecks
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
                preprocess_video=not args.no_preprocess,
                anomaly_threshold=args.anomaly_threshold,
                stampede_threshold=args.stampede_threshold,
                max_bottlenecks=args.max_bottlenecks
            )
        
        if result:
            print(f"Processing completed successfully. Output: {result}")
            
            # Launch panic simulation if requested
            if args.run_panic_sim:
                print("Launching panic/stampede simulation...")
                
                # Check if panic_simulation.py exists
                if not os.path.exists("panic_simulation.py"):
                    print("Error: panic_simulation.py not found.")
                    return 1
                
                try:
                    # Try to import the panic simulation directly for better control
                    try:
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from panic_simulation import PanicSimulation
                        
                        # Create simulation instance
                        simulation = PanicSimulation()
                        
                        # Load data
                        if simulation.load_data_from_video(args.video_path):
                            # Auto-start recording if requested
                            if args.auto_record:
                                print(f"Auto-recording enabled for {args.record_duration} seconds")
                                simulation.start_recording(duration=args.record_duration)
                            
                            # Run simulation
                            simulation.run_simulation()
                            print("Panic simulation completed.")
                        else:
                            print(f"Failed to load data from {args.video_path}")
                            return 1
                            
                    except ImportError:
                        # Fall back to subprocess call
                        print("Using subprocess to launch panic simulation...")
                        # Cannot pass auto-record flag when using subprocess
                        if args.auto_record:
                            print("Warning: Auto-record flag will be ignored in subprocess mode")
                        
                        subprocess.run([sys.executable, "panic_simulation.py", args.video_path], 
                                    check=True)
                        print("Panic simulation completed.")
                        
                except Exception as e:
                    print(f"Error running panic simulation: {e}")
                    return 1
            
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