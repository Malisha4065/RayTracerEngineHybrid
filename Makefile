# Compiler
NVCC = nvcc

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Target executables
TARGET = $(BINDIR)/raytracer
ANALYZER_TARGET = $(BINDIR)/hybrid_analyzer

# Source files
CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Analyzer source
ANALYZER_SRC = hybrid_performance_analyzer.cpp
ANALYZER_OBJ = $(OBJDIR)/hybrid_performance_analyzer.o
ANALYZER_CUDA_OBJ = $(OBJDIR)/hybrid_analyzer_cuda.o

# Compiler flags
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_50 -Xcompiler -fopenmp
INCLUDES = -I$(INCDIR)
LIBS = -lSDL2 -lcurand -lm -lgomp

# Create directories if they don't exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Default target
all: $(TARGET) $(ANALYZER_TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(LIBS)

# Compile analyzer
$(ANALYZER_OBJ): $(ANALYZER_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile analyzer CUDA wrapper
$(ANALYZER_CUDA_OBJ): $(SRCDIR)/hybrid_analyzer_cuda.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link analyzer
$(ANALYZER_TARGET): $(ANALYZER_OBJ) $(ANALYZER_CUDA_OBJ) $(filter-out $(OBJDIR)/main.o, $(OBJECTS))
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Compile CUDA source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Rebuild everything
rebuild: clean all

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install libsdl2-dev

# Run analyzer
run-analyzer: $(ANALYZER_TARGET)
	./$(ANALYZER_TARGET)

# Run raytracer
run-raytracer: $(TARGET)
	./$(TARGET)

.PHONY: all clean rebuild install-deps run-analyzer run-raytracer