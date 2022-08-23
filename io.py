import tkinter, tkinter.filedialog as tkfiledialog



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
