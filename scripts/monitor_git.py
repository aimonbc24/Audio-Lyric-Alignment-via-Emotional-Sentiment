import os
import time
import subprocess

def git_command(command):
    """Execute a Git command and return its output."""
    try:
        output = subprocess.check_output(["git"] + command.split(), stderr=subprocess.STDOUT)
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print("Error executing Git command:", e.output.decode())

def monitor_folder(folder_path, n):
    """Monitor the folder and perform Git operations when file count increases by n."""
    prev_count = len(os.listdir(folder_path))
    while True:
        current_count = len(os.listdir(folder_path))
        
        if current_count - prev_count >= n:
            print(f"File count increased by {n}, performing Git operations...")
            # Add the new files to Git
            git_command(f"add Dataset/data/mp4/*")
            # Commit the new files
            git_command("commit -m Added_new_files")
            # Push to the origin
            git_command("push origin main")  # Adjust branch name as necessary
            
            prev_count = current_count
            print("Git operations completed. There are now", current_count, "files in the folder.")
        
        time.sleep(15)  # Check every 3 minutes seconds

if __name__ == "__main__":
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    mp4_folder = os.path.join(absolute_path, r"..\Dataset\data\mp4")

    n = 5  # Set the desired threshold for file count increase
    monitor_folder(mp4_folder, n)
