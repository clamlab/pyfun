import os
import sys
import msvcrt

class SingletonInstance:
    """Ensures only one instance of the script runs at a time."""
    def __init__(self, lock_file, script_name=None):
        self.lock_file = lock_file
        self.script_name = script_name or "Another instance"

    def acquire_lock(self):
        """Try to acquire a lock, exit if already locked."""
        try:
            self.file_handle = open(self.lock_file, 'w')
            msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print(f"{self.script_name} is already running. Exiting...")
            sys.exit(1)

    def release_lock(self):
        """Release the lock when done."""
        if self.file_handle:
            msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.file_handle.close()
            os.remove(self.lock_file)
