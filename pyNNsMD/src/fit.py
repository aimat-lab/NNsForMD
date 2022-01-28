import os
import subprocess
import sys
import logging

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def get_path_for_fit_script(fit_script):
    file_path = os.path.realpath(os.path.dirname(__file__))
    folder_sequence = os.path.split(file_path)
    filepath = os.path.join(*folder_sequence[:-1], "training")
    module_logger.info("Training scripts are located at %s" % filepath)
    return os.path.join(filepath, fit_script)


def fit_model_get_python_cmd_os():
    """Return proper commandline command for pyhton depending on os.

    Returns:
        str: Python command either python or pyhton3.

    """
    # python or python3 to run
    if sys.platform[0:3] == 'win':
        return 'python'  # or 'python.exe'
    else:
        return 'python3'


def fit_model_by_script(i, fit_script, g, filepath, m, proc_async):
    """
    Run the training script in subprocess.

    Args:
        i (int): Index of model.
        fit_script (str): Name of the training routine.
        filepath (str): Filepath to model.
        g (int): GPU index to use.
        m (str): Fitmode.
        proc_async (bool):

    Returns:
        subprocess
    """
    module_logger.info("Run: {0} for {1} of model {2} async {3}".format(fit_script, i, filepath, proc_async))
    py_script = get_path_for_fit_script(fit_script)
    py_cmd = fit_model_get_python_cmd_os()

    if py_script is None:
        module_logger.error("Empty training script. Not starting training.")
        return

    if os.path.splitext(py_script)[-1] == "":
        py_script = os.path.realpath(os.path.splitext(py_script)[0] + ".py")

    if not os.path.exists(py_script):
        module_logger.error("Wrong training script %s, please check path %s" % py_script)
        raise FileNotFoundError("Can not find training script %s, please check path" % py_script)

    if proc_async:
        proc = subprocess.Popen([py_cmd, py_script, "-i", str(i), '-f', filepath, "-g", str(g), '-m', str(m)])
        return proc

    if not proc_async:
        proc = subprocess.run([py_cmd, py_script, "-i", str(i), '-f', filepath, "-g", str(g), '-m', str(m)],
                              capture_output=False, shell=False)
        return proc
