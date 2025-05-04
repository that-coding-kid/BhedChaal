import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch BhedChaal Streamlit App')
    parser.add_argument('--advanced', action='store_true', 
                      help='Launch the advanced version of the app')
    parser.add_argument('--interactive', action='store_true',
                      help='Launch the interactive version with mouse point selection')
    parser.add_argument('--simple', action='store_true',
                      help='Launch the simple interactive version (more compatible)')
    args = parser.parse_args()
    
    print("Starting BhedChaal Streamlit App...")
    
    # Determine which app to run
    if args.simple:
        app_file = "app_simple_interactive.py"
        print("Using simple interactive mode (more compatible)")
    elif args.interactive:
        app_file = "app_interactive.py"
        print("Using interactive mode with mouse point selection")
    elif args.advanced:
        app_file = "app_advanced.py"
        print("Using advanced mode with multi-step workflow")
    else:
        app_file = "app.py"
        print("Using basic mode")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version {streamlit.__version__} detected.")
    except ImportError:
        print("Streamlit not found. Installing required dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
        print("Created temp directory for storing uploaded files and results.")
    
    # Run the selected app
    print(f"Launching {app_file}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_file])

if __name__ == "__main__":
    main() 