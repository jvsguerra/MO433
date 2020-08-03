# COMMANDS
MKDIR=mkdir -p
CP=cp
PYTHON=python
RM=rm -r -f --verbose

# FOLDERS
SRC_FOLDER=src
BUILD_FOLDER=report
BIN_FOLDER=bin
ASSETS_FOLDER=input
OUTPUT_FOLDER=output
REQ=requirements.txt

# SRCs and OBJs
MAIN=$(SRC_FOLDER)/main.py
DOWNLOAD=$(SRC_FOLDER)/utils/download.py

# BUILD COMMAND
build:
	$(MKDIR) $(OUTPUT_FOLDER)

	$(MKDIR) $(OUTPUT_FOLDER)/Real

	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Real/Adam/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Real/SGD/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Noise

	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/Adam/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Noise/SGD/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Average
	
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Average/Adam/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Average/SGD/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Complete

	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/Adam/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/StepLR
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/StepLR/images
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/StepLR/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/StepLR/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/StepLR/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Complete/SGD/ReduceLROnPlateau/images/test
	
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Noise

	$(MKDIR) $(OUTPUT_FOLDER)/Final-Noise/SGD/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Noise/SGD/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Noise/SGD/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Noise/SGD/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Noise/SGD/ReduceLROnPlateau/images/test

	$(MKDIR) $(OUTPUT_FOLDER)/Final-Average

	$(MKDIR) $(OUTPUT_FOLDER)/Final-Average/SGD/ReduceLROnPlateau
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Average/SGD/ReduceLROnPlateau/images
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Average/SGD/ReduceLROnPlateau/images/train
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Average/SGD/ReduceLROnPlateau/images/val
	$(MKDIR) $(OUTPUT_FOLDER)/Final-Average/SGD/ReduceLROnPlateau/images/test


	$(MKDIR) $(OUTPUT_FOLDER)/samples

download:
	$(MKDIR) $(ASSETS_FOLDER)
	$(MKDIR) $(ASSETS_FOLDER)/dataset
	pip3 install gdown
	python3 $(DOWNLOAD)

exec:
	# Install python2 dependencies
	pip2 install opencv-python scikit-image sklearn tk
	# Prepare dataset
	python2 $(SRC_FOLDER)/utils/unsup_labels/unsup_labels.py
	# Install python3 dependencies
	pip3 install -r $(SRC_FOLDER)/$(REQ)
	# Run project
	python3 $(MAIN)

runall: build download exec

# CLEAN COMMAND
clean:
	$(RM) $(BIN_FOLDER)
	$(RM) $(OUTPUT_FOLDER)
