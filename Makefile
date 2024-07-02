# # IDIR=./
# CXX = g++

# all: clean build

# build: cudnn_example.cpp
# 	$(CXX) cudnn_example.cpp --std c++17 -o cudnn_example.exe -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -lcuda -lcudnn

# run:
# 	./cudnn_example.exe $(ARGS)

# clean:
	# rm -f cudnn_example.exe output*.txt 



# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: cuda_tiny_model.cu
	$(CXX) cuda_tiny_model.cu  --extended-lambda --std c++17 `pkg-config opencv --cflags --libs` -o cuda_tiny_model -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda  -lcudnn -lcutensor -lcublas

run:
	./cuda_tiny_model $(ARGS)

clean:
	rm -f cuda_tiny_model embedding *log.txt