import os
import tempfile
import logging
import shutil
import ffmpeg
import subprocess
import uuid
from pathlib import Path

class VideoPreprocessor:
    """
    Class for preprocessing videos using ffmpeg to standardize resolution and format.
    This helps prevent issues with extremely high-resolution videos that might cause
    performance problems or unpredictable behavior in the visualization pipeline.
    """
    def __init__(self, target_resolution=(1280, 720), output_format='mp4', temp_dir=None):
        """
        Initialize the video preprocessor.
        
        Parameters:
        - target_resolution: Tuple of (width, height) for the standardized resolution
        - output_format: Output video format (mp4, avi, etc.)
        - temp_dir: Directory to store temporary processed videos (None for system default)
        """
        self.target_resolution = target_resolution
        self.output_format = output_format
        
        # Create temp directory for processed videos if not provided
        if temp_dir is None:
            self.temp_dir = os.path.join(tempfile.gettempdir(), 'video_preprocessor')
        else:
            self.temp_dir = temp_dir
            
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('VideoPreprocessor')
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            self.logger.info("ffmpeg found and ready to use")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("ffmpeg not found! Please install ffmpeg and make sure it's in your PATH")
            raise RuntimeError("ffmpeg not found! Video preprocessing will not work.")
    
    def process_video(self, input_path):
        """
        Process a video to standardize its resolution.
        
        Parameters:
        - input_path: Path to the input video file
        
        Returns:
        - Path to the processed video file
        """
        if not os.path.exists(input_path):
            self.logger.error(f"Input video not found: {input_path}")
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create a unique filename for the processed video
        input_filename = os.path.basename(input_path)
        input_name = os.path.splitext(input_filename)[0]
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{input_name}_processed_{unique_id}.{self.output_format}"
        output_path = os.path.join(self.temp_dir, output_filename)
        
        self.logger.info(f"Processing video: {input_path}")
        self.logger.info(f"Target resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        
        try:
            # Get video info
            probe = ffmpeg.probe(input_path)
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            
            if video_stream is None:
                self.logger.error(f"No video stream found in {input_path}")
                raise ValueError(f"No video stream found in {input_path}")
            
            # Get original width and height
            orig_width = int(video_stream['width'])
            orig_height = int(video_stream['height'])
            
            self.logger.info(f"Original resolution: {orig_width}x{orig_height}")
            
            # Skip processing if resolution is already lower than target
            if orig_width <= self.target_resolution[0] and orig_height <= self.target_resolution[1]:
                self.logger.info("Video resolution already below target, skipping processing")
                return input_path
            
            # Process video with ffmpeg-python
            (
                ffmpeg
                .input(input_path)
                .output(output_path, 
                       vf=f"scale={self.target_resolution[0]}:{self.target_resolution[1]}:force_original_aspect_ratio=decrease,pad={self.target_resolution[0]}:{self.target_resolution[1]}:(ow-iw)/2:(oh-ih)/2",
                       **{'crf': '22', 'preset': 'medium'})
                .run(quiet=True, overwrite_output=True)
            )
            
            self.logger.info(f"Video processed and saved to: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            self.logger.error(f"Error processing video: {e.stderr.decode() if e.stderr else str(e)}")
            self.logger.info("Returning original video path")
            return input_path
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.info("Returning original video path")
            return input_path
    
    def cleanup(self):
        """Remove all temporary processed videos"""
        try:
            self.logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            self.logger.info("Cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 