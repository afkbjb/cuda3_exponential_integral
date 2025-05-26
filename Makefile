CC        = g++
CFLAGS    = -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -I.

NVCC      = nvcc
NVCCFLAGS = -std=c++11 -O3 -lineinfo -I.

EXEC      = exponentialIntegral.out

SRC_CPP   = cpu.cpp
SRC_CU    = gpu.cu main.cu
OBJ_CPP   = $(SRC_CPP:.cpp=.o)
OBJ_CU    = $(SRC_CU:.cu=.o)

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ_CPP) $(OBJ_CU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cpp ei.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu ei.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) $(OBJ_CPP) $(OBJ_CU)
