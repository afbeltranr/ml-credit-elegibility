# Makefile

# Default target when 'make' is run
all: install run

# Install Python dependencies
install:
	pip install -r requirements.txt

# Run the logistic model and output the metrics
run:
	python train_logistic_model.py

# Clean up (optional)
clean:
	rm -rf __pycache__