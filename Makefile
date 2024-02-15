# Check if the OS is Windows
VIRTUAL_ENV := .venv
ifeq ($(OS),Windows_NT)
    VIRTUAL_ENV_BIN := $(VIRTUAL_ENV)\Scripts
    ACTIVATE_VENV := $(VIRTUAL_ENV_BIN)\activate
else
    VIRTUAL_ENV_BIN := $(VIRTUAL_ENV)/bin
    ACTIVATE_VENV := $(VIRTUAL_ENV_BIN)/activate
endif

.PHONY: all detect dold ffmpeg

all: activate
	@echo "Running main.py"
	$(VIRTUAL_ENV_BIN)/python ./src/main.py

detect: activate
	@echo "Running detector.py"
	$(VIRTUAL_ENV_BIN)/python ./src/detector.py -O100

ana:
	@echo "Analyzing.."
	$(VIRTUAL_ENV_BIN)/python analyze_profiled.py

dold: activate
	@echo "Running detector_old_assuming_16_9_ratio.py"
	$(VIRTUAL_ENV_BIN)/python ./src/detector_old_assuming_16_9_ratio.py

ffmpeg: activate
	@echo "Running ffmpeg.py"
	$(VIRTUAL_ENV_BIN)/python ./src/ffmpeg.py

# Create virtual environment if not exists
activate: $(ACTIVATE_VENV)

$(ACTIVATE_VENV):
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VIRTUAL_ENV)" ]; then \
	    python3 -m venv $(VIRTUAL_ENV); \
	    echo "Installing Dependencies"; \
	    $(VIRTUAL_ENV_BIN)/python3 -m pip install -r requirements.txt; \
	fi
