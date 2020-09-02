"""
Main interface to start training_??.py scripts in parallel. This can be solved in many different ways.
Possible are server solutions with slurm and MPI. Here only python subprocess are started to local machine.
The training scripts are supposed to read all necessary information from folder. 
NOTE: Path information of folder and training scripts as well as os info are made fetchable but could fail in certain
circumstances.
"""
import numpy as np
import time
import os
import sys
import subprocess
import sys


def get_path_for_fit_script():
    """
    Function to find the path of training scripts. 
    For now they are expected to be in the same folder as calling .py script.

    Returns
    -------
    filepath : str
        Filepath pointing to training scripts.

    """
    #Ways of finding path either os.getcwd() or __file__ or just set static path with install...
    locdiR = os.getcwd()
    filepath = os.path.dirname(__file__) 
    STATIC_PATH_FIT_SCRIPT = ""
    return filepath


def fit_model_get_python_cmd_os():
    """
    Return proper commandline command for pyhton depending on os.

    Returns
    -------
    str
        Python command either python or pyhton3.

    """
    # python or python3 to run
    if(sys.platform[0:3] == 'win'):
        return 'python' # or 'python.exe'
    else:
        return 'python3'


def fit_model_energy_gradient(i,filepath,g,m):
    """
    Run the training script in subprocess.

    Parameters
    ----------
    i : int
        Index of model.
    filepath : str
        Filepath to model.
    g : int
        GPU index to use.
    m : str
        Fitmode.

    Returns
    -------
    None.

    """
    print("Run:",filepath,"Instance:",i, "on GPU:",g,m)
    py_cmd = fit_model_get_python_cmd_os()
    py_script = os.path.join(get_path_for_fit_script(),"training_eg.py")
    if(os.path.exists(py_script) == False):
        print("Error: Can not find trainingsript, please check path")
    subprocess.run([py_cmd,py_script, "-i",str(i),'-f',filepath,"-g",str(g),'-m',str(m)],capture_output=False,shell = False)
    #proc_out = str(proc.stdout.decode('utf8'))
    #print(proc_out)
    return


def fit_model_nac(i,filepath,g,m):
    """
    Run the training script in subprocess.

    Parameters
    ----------
    i : int
        Index of model.
    filepath : str
        Filepath to model.
    g : int
        GPU index to use.
    m : str
        Fitmode.

    Returns
    -------
    None.

    """
    print("Run:",filepath,"Instance:",i, "on GPU:",g,m)
    py_cmd = fit_model_get_python_cmd_os()
    py_script = os.path.join(get_path_for_fit_script(),"training_nac.py")
    if(os.path.exists(py_script) == False):
        print("Error: Can not find trainingsript, please check path")
    subprocess.run([py_cmd,py_script,"-i",str(i),'-f',filepath,"-g",str(g),'-m',str(m)],capture_output=False,shell = False) 
    #proc_out = str(proc.stdout.decode('utf8'))
    #print(proc_out)
    return
        

def fit_model_energy_gradient_async(i,filepath,g,m):
    """
    Run the training script in subprocess.

    Parameters
    ----------
    i : int
        Index of model.
    filepath : str
        Filepath to model.
    g : int
        GPU index to use.
    m : str
        Fitmode.

    Returns
    -------
    subprocess.Popen.

    """
    print("Run:",filepath,"Instance:",i, "on GPU:",g,m)
    py_cmd = fit_model_get_python_cmd_os()
    py_script = os.path.join(get_path_for_fit_script(),"training_eg.py")
    if(os.path.exists(py_script) == False):
        print("Error: Can not find trainingsript, please check path")
    proc = subprocess.Popen([py_cmd,py_script,"-i",str(i),'-f',filepath,"-g",str(g),'-m',str(m)]) 
    return proc


def fit_model_nac_async(i,filepath,g,m):
    """
    Run the training script in subprocess.

    Parameters
    ----------
    i : int
        Index of model.
    filepath : str
        Filepath to model.
    g : int
        GPU index to use.
    m : str
        Fitmode.

    Returns
    -------
    subprocess.Popen.

    """
    print("Run:",filepath,"Instance:",i, "on GPU:",g,m)
    py_cmd = fit_model_get_python_cmd_os() 
    py_script = os.path.join(get_path_for_fit_script(),"training_nac.py")
    if(os.path.exists(py_script) == False):
        print("Error: Can not find trainingsript, please check path")
    proc = subprocess.Popen([py_cmd,py_script,"-i",str(i),'-f',filepath,"-g",str(g),'-m',str(m)])
    return proc