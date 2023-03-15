import os
import shutil
import datetime
from tqdm import tqdm
import time

"""
EC march 2023
usage: python make_backup.py

simple python script to back up data. Briefly looked at other automatic software solutions,
but one issue with these, is how they handle deleted files in source folder 
(they may attempt to sync destination folder, and also delete the source file). 
"""



# Set the source and destination paths
X = '/path/to/source/folder'
Y = '/path/to/destination/folder'

# Set the retry count and sleep time--how many times to try to copy a file if it fails
retry_count = 3
sleep_time = 1 #in seconds, time to wait between re-tries

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
            # Check if the file already exists in Y
            if os.path.exists(dst_path):
                # Get the size and modification time of both files
                src_size = os.path.getsize(src_path)
                src_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(src_path))
                dst_size = os.path.getsize(dst_path)
                dst_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(dst_path))
                # Compare the sizes and modification times
                if src_size > dst_size or src_mtime > dst_mtime:
                    # Retry copying up to retry_count times with a sleep_time delay in between
                    retry = 0
                    while retry < retry_count:
                        try:
                            shutil.copy2(src_path, dst_path)
                            pbar.update(1)
                            break
                        except:
                            retry += 1
                            print(f"Connection lost while copying {file}. Retrying ({retry}/{retry_count})...")
                            time.sleep(sleep_time)
                            continue
                else:
                    # Print information about the conflicting files
                    print(f"File {file} already exists in {Y} with the same size and modification time:")
                    print(f"\t{src_path} - {src_size} bytes, modified {src_mtime}")
                    print(f"\t{dst_path} - {dst_size} bytes, modified {dst_mtime}")
            else:
                # Retry copying up to retry_count times with a sleep_time delay in between
                retry = 0
                while retry < retry_count:
                    try:
                        shutil.copy2(src_path, dst_path)
                        pbar.update(1)
                        break
                    except:
                        retry += 1
                        print(f"Connection lost while copying {file}. Retrying ({retry}/{retry_count})...")
                        time.sleep(sleep_time)
                        continue
