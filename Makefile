# Makefile for building a custom local docker image and launching a JupyterLab from the containerized image 
# Docker image can be built by running "make build" in shell (DO THIS ONLY ONCE)
# Run "make local" to open JupyterLab from containerized image 

# Name to give the image 
IMG_NAME = era-carb

# "make local"
# Launch local JupyerLab from containerized instance of IMG_NAME
# Open JupyterLab by clicking the bottom link in the terminal output 
local:
	@echo "Opening JupyterLab from runtime instance of $(IMG_NAME)"
	@echo "Copy/paste the last link in the terminal output into your web browser to open JupyterLab"
	docker run -t --rm --volume "$(PWD)":/home/jovyan \
	-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	-p 8888:8888 $(IMG_NAME) jupyter lab --ip 0.0.0.0

# "make build"
# YOU ONLY NEED TO DO THIS ONCE 
# Create the docker image locally using the Dockerfile in the current repository 
# Output from this command can be found in the logfile build.log
build: 
	@echo "Building docker image named $(IMG_NAME) from Dockerfile"
	@echo "A conda environment will be created in the image using a conda-lock.yml file if it exists. If not, the environment.yml will be used."
	@echo "Pip dependencies will be installed from the requirements.txt if it exists."
	@echo "Output from the build will be saved in a log file build.log"
	docker build -t $(IMG_NAME) .  &> build.log

# "make conda-lock" 
# Make conda-lock.yml from environment.yml 
# You must have conda-lock installed 
conda-lock: 
	@echo "Creating multi-platform conda-lock.yml from environment.yml"
	@echo "Will fail if conda-lock package is not installed"
	conda-lock -f environment.yml -p osx-64 -p linux-64

# BONUS: "make run" 
# Run a python script in a runtime instance (container) of the image 
# Customize the path below to add your own script 
# Add any additional arguments for python script to ARGS
run: 
	@echo "Running python script $(PY_SCRIPT) in runtime instance of $(IMG_NAME)"
	docker run -t --rm --volume "$(PWD)":/home/jovyan \
	-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	$(IMG_NAME) python $(PY_SCRIPT) $(ARGS)
