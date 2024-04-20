import tkinter, tkinter.filedialog as tkfiledialog

import os
import shutil



def copy_file(source_file: str, destination: str, overwrite: bool = False) -> None:
    """
    Copies a file from the source folder to a destination folder or file, creating all necessary parent folders if they don't exist.

    Parameters:
        source_file (str): The path of the source file to copy.
        destination (str): The path of the destination folder or file to copy the file to. If the destination is a folder, the function preserves the original filename of the source file. If the destination is a file, the function uses the specified filename.
        overwrite (bool): If True, overwrites the destination file if it already exists. If False, raises a warning and skips the copy operation if the destination file already exists.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        shutil.Error: If the copy operation fails for any reason.
    """
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file '{source_file}' not found.")

    if os.path.isdir(destination):
        filename = os.path.basename(source_file)
        destination_file = os.path.join(destination, filename)
    else:
        destination_file = destination
        filename = os.path.basename(destination_file)

    if os.path.exists(destination_file):
        if overwrite:
            print(f"Warning: File '{destination_file}' already exists and will be overwritten.")
        else:
            print(f"Warning: File '{destination_file}' already exists. Skipping copy operation.")
            return

    try:
        # Create the destination folder and all parent folders if they don't exist
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)

        # Copy the file from source to destination with new filename
        shutil.copy(source_file, destination_file)

    except shutil.Error as e:
        raise shutil.Error(f"Error copying file: {e}")


def get_file(initial_dir=None, title=''):
    root = tkinter.Tk()
    root.lift()
    root.after(100, root.focus_force)
    root.after(200, root.withdraw)
    f_path = tkfiledialog.askopenfilename(parent=root, initialdir=initial_dir, title=title)

    return f_path


def get_folder(initial_dir=None, title=''):
    root = tkinter.Tk()
    root.lift()
    root.after(100, root.focus_force)
    root.after(200, root.withdraw)
    fd_path = tkfiledialog.askdirectory(parent=root, initialdir=initial_dir, title=title)

    return fd_path


def generate_unique_filename(filename, overwrite=False):
    """
    Generates a unique filename by appending a counter before the file extension.
    If overwrite is True, returns the original filename.

    :param filename: Original filename
    :param overwrite: Whether to overwrite existing files (default False)
    :return: Unique filename
    """
    if overwrite or not os.path.exists(filename):
        return filename

    # Split the filename into name and extension
    name_part, extension = os.path.splitext(filename)
    counter = 1

    # Generate a new filename by appending a counter before the extension
    new_filename = f"{name_part}_{counter:02}{extension}"

    # Increment the counter until a unique filename is found
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{name_part}_{counter:02}{extension}"

    return new_filename