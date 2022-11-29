# Instruction on how to run RCI cluster jobs

# Connect to the cluster
ssh kretoant@login.rci.cvut.cz

# enter the password (librarianisHERE1998!new)

# enter root directory of the project
# This project has to be already cloned from git
# If not, do this:

# git clone https://github.com/DevKretov/ntu_nlp_al.git
# git checkout rci_cluster_integration
# git pull

cd ntu_nlp_al

# Load the environment
# First, load all system GPU dependencies inside RCI CLUSTER

# For optimization of PyTorch (required)
module load OpenBLAS/0.3.20-GCC-11.3.0

# For PyTorch native support in order to be able to utilise GPU
ml PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

# NOTE: CUDA version must be in two first parts of version the same as OpenBLAS, otherwise the system won't be able to pair them and thus import torch will fail
# In case you don't know what module to load, use this:

# module spider <module_name>
# <module_name> can be "PyTorch/1.10.0-foss-2021a-CUDA-11.3.1", can be just "pytorch" or etc. It will show all available modules with this substring

# Then, we need to create virtual environment


# create virtualenv with the name venv_2
virtualenv venv_2
# activate it 
source venv_2/bin/activate

# in case you want to deactivate it, use
# deactivate


# Then, install all requirements right from requirements.txt file:
python -m pip install -r requirements.txt --upgrade

# ALL requirements MUST have their exact version specified!!! Like torch==1.10.1, otherwise I cannot guarantee what will happen

# Go to the .batch file location in order to run it 
# NOTE: you MUST be in this directory, otherwise rewrite .batch file (it's literally a shell script)
cd slurm/

# This will run GPU job basing on parameters you set in this file
# The most important one is that you MUST have the path to the logging file(s) existent. The directory where these files are created has to exist beforehand!!!
sbatch train_gpu.batch 

# To see your job running on RCI you can use 
squeue

# Or, in smarter way:
squeue | grep <your_job_id>

# if you want to see the result of the run, go to Wandb.ai (reccommend to use it)
# of you can go to <your_logs_directory>, then do

# cd <your_logs_directory>

# more <log_file_name>



# That's it! Have fun!
