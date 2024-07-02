# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: src/cuda_tiny_model.cu
	$(CXX) src/cuda_tiny_model.cu  --extended-lambda --std c++17 `pkg-config opencv --cflags --libs` -o ./bin/cuda_tiny_model -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda  -lcudnn -lcutensor -lcublas

run:
	./bin/cuda_tiny_model $(ARGS)

clean:
	rm -f ./bin/cuda_tiny_model *log.txt