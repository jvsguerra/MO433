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

# BUILD COMMAND
build:
	$(MKDIR) $(OUTPUT_FOLDER)
	$(MKDIR) $(OUTPUT_FOLDER)/losses
	$(MKDIR) $(OUTPUT_FOLDER)/bpd
	$(MKDIR) $(OUTPUT_FOLDER)/samples
	$(MKDIR) $(OUTPUT_FOLDER)/cifar
	$(MKDIR) $(OUTPUT_FOLDER)/dogs_cifar

exec:
	pip3 install -r $(SRC_FOLDER)/$(REQ)
	python3 $(MAIN)

runall: build exec

# CLEAN COMMAND
clean:
	$(RM) $(BIN_FOLDER)
	$(RM) $(OUTPUT_FOLDER)
