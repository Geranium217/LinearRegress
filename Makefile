# Makefile for the linear regression project

# Compiler flags
CFLAGS = -g -O2 -I/usr/local/cuda/include

# CUDA compiler flags
CUDAFLAGS = -g -O2 -I/usr/local/cuda/include

# Linker flags
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

# Source files
SRCS = linear_regression.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = linear_regression

# Build the project
all: $(TARGET)

# Compile the source files
$(TARGET): $(OBJS)
  $(CXX) $(LDFLAGS) -o $@ $^

# Compile the CUDA source files
%.o: %.cpp
  $(NVCC) $(CUDAFLAGS) -o $@ $<

# Clean the project
clean:
  rm -f $(OBJS) $(TARGET)
