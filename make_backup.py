import os
import shutil
import datetime
from tqdm import tqdm
import time
import sys

"""
EC march 2023
usage: python make_backup.py

simple python script to back up data. Briefly looked at other automatic software solutions,
but one issue with these, is how they handle deleted files in source folder 
(they may attempt to sync destination folder, and also delete the source file). 
"""

# Set the source and destination paths
X = 'D:/Data'
Y = 'Y:/Edmund/Data/Touchscreen_pSWM'


# Set the retry count and sleep time
retry_count = 3
sleep_time = 1


def copy_file():
    retry = 0
    while retry < retry_count:
        try:
            shutil.copy2(src_path, dst_path)
            pbar.update(1)
            return True
        except:
            retry += 1
            print(f"Connection lost while copying {file}. Retrying ({retry}/{retry_count})...")
            time.sleep(sleep_time)
            continue
    return False


# Check if X or Y exists, or else exit
for fd in [X,Y]:
    if not os.path.exists(fd):
        print(f"Source or destination fikder {fd} does not exist.")
        sys.exit()

# Get the total number of files to be copied
total_files = sum([len(files) for r, d, files in os.walk(X)])

# Initialize the progress bar
with tqdm(total=total_files) as pbar:
    # Iterate over all files and subdirectories in X
    for root, dirs, files in os.walk(X):
        # Create the corresponding subdirectories in Y
        for dir in dirs:
            os.makedirs(os.path.join(Y, root.replace(X, ''), dir), exist_ok=True)
        # Copy each file to Y, handling conflicts and retrying on connection loss
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(Y, root.replace(X, ''), file)

            # Print the current file being copied (overwrite previous line)
            sys.stdout.write(f"\rCopying {file}...")
            sys.stdout.flush()            
            
            # Check if the file already exists in Y
            if os.path.exists(dst_path):
                # Get the size and modification time of both files
                src_size = os.path.getsize(src_path)
                src_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(src_path))
                dst_size = os.path.getsize(dst_path)
                dst_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(dst_path))

                # Compare the sizes
                if src_size > dst_size: #replace destination file if smaller

                    success = copy_file()

                    while not success:
                        print(f"Failed to copy {file} after {retry_count} retries. Skipping...")
                        break

                    # Print information about the replaced file
                    print(f"Old file was replaced with new:")
                    print(f"\t{src_path} - {src_size} bytes, modified {src_mtime}")
                    print(f"\t{dst_path} - {dst_size} bytes, modified {dst_mtime}")


            else:
                success = copy_file()

                while not success:
                    print(f"Failed to copy {file} after {retry_count} retries. Skipping...")
                    break

