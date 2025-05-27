# Makefile for CUDA Exponential Integral Calculator

# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11
CUDA_LIBS = -lcuda -lcudart

# Target executable
TARGET = llm

# Source files
SOURCES = llm.cu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES) $(CUDA_LIBS)

# Clean target
clean:
	rm -f $(TARGET)

# Install CUDA (Ubuntu/Debian example)
install-cuda:
	@echo "To install CUDA on Ubuntu/Debian:"
	@echo "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"
	@echo "sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
	@echo "wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb"
	@echo "sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb"
	@echo "sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/"
	@echo "sudo apt-get update"
	@echo "sudo apt-get -y install cuda"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the CUDA exponential integral calculator"
	@echo "  clean        - Remove built files"
	@echo "  install-cuda - Show CUDA installation instructions"
	@echo "  help         - Show this help message"

.PHONY: all clean install-cuda help