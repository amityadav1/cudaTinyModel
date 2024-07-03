#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <tuple>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cutensor.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>


const int block_size = 16; // T
const int batch_size = 32; // B
const int n_embed = 64;  // C
const int n_hidden = 100; // H
int vocab_size = 0; // V
const int max_steps = 120;
float learning_rate = 0.0001f;

const int threadsPerBlock = 256;


void testConvolution();
void checkDataCopyToGPU(int *Xtr, int Xtr_size, int *Ytr, int *d_Xtr, int *d_Ytr, std::map<int, char>& itos);
void printRandomIndices(int *d_idx, std::map<int, char>& itos);

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess) {                                          \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";   \
            std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                         \
        }                                                                    \
    }

#define CHECK_CUDNN(call)                                                    \
    {                                                                        \
        const cudnnStatus_t error = call;                                    \
        if (error != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";   \
            std::cerr << "code: " << error << ", reason: " << cudnnGetErrorString(error) << std::endl; \
            exit(1);                                                         \
        }                                                                    \
    }


// Random number generator kernel
__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}


// Kernel to generate a normal distribution for random initialization. 
__global__ void generate_normal_kernel(float *data, curandState *state, int size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size) {
        data[id] = curand_normal(&state[id]);
    }
}


//  kernel to generate random indices for minibatch index sampling
__global__ void generate_indices_kernel(int *indices, curandState *state, int batch_size, int range) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        indices[id] = curand(&state[id]) % range;
    }
}

// CUDA kernel to generate minibatch
__global__ void generate_training_minibatch(int *d_X, int *d_Y, int *d_Xtr, int *d_Ytr, int *d_idx, int batch_size, int block_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        for (int i  = 0 ; i < block_size; i++) {
            d_X[id * block_size + i] = d_Xtr[d_idx[id] * block_size + i];
        }
        d_Y[id] = d_Ytr[id];
    }
}

// CUDA kernel to generate Input embeddings corresponding to the minibatch
// CUDA and CUDNN does not support embeddings natively has embeddings are handled explicitly
__global__ void lookup_embeddings(float *d_embd, int *d_X, float *d_Xembd, int batch_size, int block_size, int n_embd) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < n_embd; j++) {
                int outidx = id * block_size * n_embd + i * n_embd + j;
                d_Xembd[outidx] = d_embd[d_X[id * block_size + i] * n_embd + j];
            }
        }
    }
}

// Kernel to compute cross entropy loss as a loss function.
__global__ void crossEntropyLossKernel(float* output, int* labels, int batchSize, int numClasses, float* loss) {
    float localLoss = 0.0f;
    for (int i = 0; i < batchSize; ++i) {
        int label = labels[i];
        if (label >= 0 && label < numClasses) {
            localLoss -= logf(output[i * numClasses + label]);
        }
    }
    *loss = localLoss / batchSize;
}


__global__ void adjustGradientsKernel(float* gradients, int* labels, int batchSize, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        int label = labels[idx];
        if (label >= 0 && label < numClasses) {
            gradients[idx * numClasses + label] -= 1.0f;
        }
    }
}

__global__ void updateParametersKernel(float* params, float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grads[idx];
    }
}

// Compute gradiensts for embeddings
__global__ void computeEmbeddingGradients(float *d_dXembd, int *d_X, float *d_dEmbd, 
                                          int batch_size, int block_size, int n_embed, int vocab_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < batch_size * block_size; i += stride) {
        int batch_idx = i / block_size;
        int seq_idx = i % block_size;
        int token = d_X[i];

        for (int j = 0; j < n_embed; j++) {
            atomicAdd(&d_dEmbd[token * n_embed + j], 
                      d_dXembd[batch_idx * block_size * n_embed + seq_idx * n_embed + j]);
        }
    }
}


// Find the character with the max probability
__global__ void findMaxIndex(float* array, int size, int* maxIndex) {
    if (threadIdx.x == 0) {  // Only the first thread does the work
        float maxVal = array[0];
        int maxIdx = 0;

        for (int i = 1; i < size; ++i) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }

        *maxIndex = maxIdx;
    }
}


// Function to initialize CUDA memory with random numbers
__host__ void initialize_random(float *data, int size, float scale, float bias) {
    curandState *devStates;
    cudaMalloc((void **)&devStates, size * sizeof(curandState));
    setup_kernel<<<(size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(devStates, 2147483647);
    generate_normal_kernel<<<(size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(data, devStates, size);
    cudaFree(devStates);

    // Scale and bias adjustment
    thrust::device_ptr<float> dev_ptr(data);
    thrust::transform(dev_ptr, dev_ptr + size, dev_ptr, [=] __device__ (float x) { return x * scale + bias; });
}


// Input data preperation function. The function reads names 
// from the input file one name per line and returns them as
// vectors.
__host__ std::vector<std::string> readAndSplitLines(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> lines;

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();
    return lines;
}



// To create a vocabulary for the tokens (characters in this case), Prepare
// maps which maps each character to a index and vice versa.
// stoi mapping would map each character to a integer (index)
// itos is the reverse mapping of that.
__host__ void createMappings(const std::vector<char>& chars, std::map<char, int>& stoi, std::map<int, char>& itos) {
    for (size_t i = 0; i < chars.size(); ++i) {
        stoi[chars[i]] = i+1;
        itos[i+1] = chars[i];
    }
    stoi['.'] = 0;
    itos[0] = '.';
}

// Function to encode a string - Given a string return the indices
// of the characters in the string.
__host__ std::vector<int> encode(const std::string& s, const std::map<char, int>& stoi) {
    std::vector<int> encoded;
    for (char c : s) {
        encoded.push_back(stoi.at(c));
    }
    return encoded;
}

// Function to decode a vector of integers - Given indices, return the string they represent.
__host__ std::string decode(const std::vector<int>& l, const std::map<int, char>& itos) {
    std::string decoded;
    for (int i : l) {
        decoded.push_back(itos.at(i));
    }
    return decoded;
}

// Function to split data into training and validation sets
__host__ void splitData(const std::vector<int>& data, double train_ratio, std::vector<int>& train_data, std::vector<int>& val_data) {
    size_t n = static_cast<size_t>(train_ratio * data.size());
    train_data.assign(data.begin(), data.begin() + n);
    val_data.assign(data.begin() + n, data.end());
}


// Function to build dataset for training
__host__ std::tuple<int *, int *, int> buildDataset(const std::vector<std::string>& words, const std::map<char, int>& stoi) {
    std::vector<std::vector<int>> X;
    std::vector<int> Y;
    
    for (const std::string& w : words) {
        std::vector<int> context(block_size, 0); // initialize context with zeros
        for (char ch : (w + '.')) {
            int ix = stoi.at(ch);
            X.push_back(context);
            Y.push_back(ix);
            context.erase(context.begin());
            context.push_back(ix);
        }
    }

    int *X_arr = new int[X.size() * block_size];
    int *Y_arr = new int[Y.size()];
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < block_size; j++) {
            X_arr[i * block_size + j] = X[i][j];
        }
        Y_arr[i] = Y[i];
    }

    // Note size returned here is number of input examples 
    // for training and not the memory size.
    return {X_arr, Y_arr, X.size()};
}


void printVersions() {
    int cudaRuntimeVersion;
    int cudaDriverVersion;
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
    cudaDriverGetVersion(&cudaDriverVersion);

    std::cout << "CUDA Runtime Version: " 
              << cudaRuntimeVersion / 1000 << "." 
              << (cudaRuntimeVersion % 100) / 10 << std::endl;
    std::cout << "CUDA Driver Version: " 
              << cudaDriverVersion / 1000 << "." 
              << (cudaDriverVersion % 100) / 10 << std::endl;

    size_t cudnnVersion = cudnnGetVersion();
    std::cout << "cuDNN Version: " 
              << cudnnVersion / 1000 << "." 
              << (cudnnVersion % 1000) / 100 << "." 
              << (cudnnVersion % 100) / 10 << std::endl;
}


__host__ std::string parseCommandLineArguments(int argc, char *argv[])
{
    std::cout << "Parsing CLI arguments\n";
    std::string inputFile = "data/names.txt";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputFile = value;
        }
    }
    std::cout << "input File Name: " << inputFile << "\n";
    return {inputFile};
}

__host__ void initialize_model_and_data(int **d_idx, int **d_X, int **d_Y, 
                                        float **d_Xembd, float **d_embd, 
                                        float **d_W1, float **d_b1, float **d_W2, 
                                        float **d_b2) {
    // Set up Array to hold indices for minibatches of training data
    CHECK_CUDA(cudaMalloc(d_idx, batch_size * sizeof(int)));
    
    // Set up input batch arrays
    CHECK_CUDA(cudaMalloc(d_X, batch_size * block_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(d_Y, batch_size * sizeof(int)));
    
    // Set up input batch size for embeddings
    CHECK_CUDA(cudaMalloc(d_Xembd, batch_size * block_size * n_embed * sizeof(float)));
    
    // Initialize Model
    // Embedding matrix
    CHECK_CUDA(cudaMalloc(d_embd, vocab_size * n_embed * sizeof(float)));
    initialize_random(*d_embd, vocab_size * n_embed, 1.0f, 0.0f);
    
    // Hidden layer weights and biases
    CHECK_CUDA(cudaMalloc(d_W1, n_hidden * block_size * n_embed * sizeof(float)));
    initialize_random(*d_W1, n_embed * block_size * n_hidden, (5.0f / 3.0f) / sqrtf(n_embed * block_size), 0.0f);
    CHECK_CUDA(cudaMalloc(d_b1, n_hidden * sizeof(float)));
    initialize_random(*d_b1, n_hidden, 0.01f, 0.0f);
    
    // Softmax layer weights and biases
    CHECK_CUDA(cudaMalloc(d_W2, n_hidden * vocab_size * sizeof(float)));
    initialize_random(*d_W2, n_hidden * vocab_size, 0.01f, 0.0f);
    CHECK_CUDA(cudaMalloc(d_b2, vocab_size * sizeof(float)));
    initialize_random(*d_b2, vocab_size, 0.0f, 0.0f);
}

__host__ void allocate_intermediate_buffers(
    float **d_Y1, float **d_Y2, 
    float **d_dW1, float **d_db1, float **d_dW2, float **d_db2, 
    float **d_dY1, float **d_dY2, float **d_dEmbd) {
    // Layer 1
    CHECK_CUDA(cudaMalloc(d_Y1, n_hidden * batch_size * sizeof(float)));

    // Layer 2
    CHECK_CUDA(cudaMalloc(d_Y2, vocab_size * n_hidden * sizeof(float)));

    // Gradients
    CHECK_CUDA(cudaMalloc(d_dW1, n_hidden * block_size * n_embed * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_db1, n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_dW2, vocab_size * n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_db2, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_dY1, batch_size * n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_dY2, batch_size * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(d_dEmbd, vocab_size * n_embed * sizeof(float)));
}


void initialize_cudnn_descriptors(
    cudnnTensorDescriptor_t *x_desc, cudnnTensorDescriptor_t *y_desc, cudnnTensorDescriptor_t *b1_desc,
    cudnnFilterDescriptor_t *w1_desc,
    cudnnTensorDescriptor_t *y2_desc, cudnnTensorDescriptor_t *b2_desc,
    cudnnFilterDescriptor_t *w2_desc,
    cudnnTensorDescriptor_t *labels_desc,
    cudnnConvolutionDescriptor_t *conv_desc,
    cudnnActivationDescriptor_t *activationDesc) {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(b1_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(w1_desc));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(y2_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(b2_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(w2_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(labels_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*labels_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, vocab_size, 1, 1));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, block_size * n_embed, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, n_hidden, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*b1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n_hidden, 1, 1));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(*w1_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n_hidden, block_size * n_embed, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*b2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, vocab_size, 1, 1));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(*w2_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, vocab_size, n_hidden, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*y2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, vocab_size, 1, 1));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(*conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Create activation descriptor
    CHECK_CUDNN(cudnnCreateActivationDescriptor(activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(*activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));
}


void forward_pass(cudnnHandle_t cudnnHandle,
                float *d_Xembd, float *d_W1, float *d_b1, float *d_W2, float *d_b2,
                float *d_Y1, float *d_Y2,
                cudnnTensorDescriptor_t x_desc, cudnnFilterDescriptor_t w1_desc,
                cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t y_desc,
                cudnnTensorDescriptor_t b1_desc, cudnnActivationDescriptor_t activationDesc,
                cudnnFilterDescriptor_t w2_desc, cudnnTensorDescriptor_t y2_desc,
                cudnnTensorDescriptor_t b2_desc,
                void *workSpace, size_t workspaceSize) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Layer 1: Convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, x_desc, d_Xembd, w1_desc, d_W1, conv_desc,
                                        CUDNN_CONVOLUTION_FWD_ALGO_GEMM, workSpace, workspaceSize, &beta, y_desc, d_Y1));
    
    // Add bias
    CHECK_CUDNN(cudnnAddTensor(cudnnHandle, &alpha, b1_desc, d_b1, &alpha, y_desc, d_Y1));

    // Apply tanh activation
    CHECK_CUDNN(cudnnActivationForward(cudnnHandle, activationDesc, &alpha, y_desc, d_Y1, &beta, y_desc, d_Y1));

    // Layer 2: Convolution (equivalent to matrix multiplication Y1 * W2)
    CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, y_desc, d_Y1, w2_desc, d_W2, conv_desc,
                                        CUDNN_CONVOLUTION_FWD_ALGO_GEMM, workSpace, workspaceSize, &beta, y2_desc, d_Y2));
    
    // Add bias
    CHECK_CUDNN(cudnnAddTensor(cudnnHandle, &alpha, b2_desc, d_b2, &alpha, y2_desc, d_Y2));

    // Apply softmax
    CHECK_CUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha, y2_desc, d_Y2, &beta, y2_desc, d_Y2));
}

__host__ float compute_cross_entropy_loss(float *d_Y2, int *d_Y, float *d_loss) {
    float h_loss = 0;  // Host variable to store the final loss

    // Launch the kernel
    dim3 block(1);
    dim3 grid(1);
    crossEntropyLossKernel<<<grid, block>>>(d_Y2, d_Y, batch_size, vocab_size, d_loss);
    CHECK_CUDA(cudaGetLastError());

    // Copy the loss from device to host
    CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

    // Print the loss
    std::cout << "Cross-Entropy Loss: " << h_loss << std::endl;

    return h_loss;
}

__host__ void backpropagation_and_update(
                cudnnHandle_t cudnnHandle,
                float *d_Y2, float *d_dY2, float *d_Y1, float *d_dY1, float *d_Xembd, int *d_X, int *d_Y,
                float *d_W1, float *d_b1, float *d_W2, float *d_b2, float *d_embd,
                float *d_dW1, float *d_db1, float *d_dW2, float *d_db2, float *d_dEmbd,
                cudnnTensorDescriptor_t y2_desc, cudnnTensorDescriptor_t y_desc, cudnnTensorDescriptor_t x_desc,
                cudnnFilterDescriptor_t w2_desc, cudnnFilterDescriptor_t w1_desc,
                cudnnTensorDescriptor_t b2_desc, cudnnTensorDescriptor_t b1_desc,
                cudnnConvolutionDescriptor_t conv_desc,
                cudnnActivationDescriptor_t activationDesc,
                void *workSpace, size_t workspaceSize) {

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int threadsPerBlock = 256;  // Adjust as needed

    // 1. Compute gradients for output layer
    CHECK_CUDNN(cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha, y2_desc, d_Y2, y2_desc, d_Y2, &beta, y2_desc, d_dY2));

    // Adjust d_dY2 based on true labels
    adjustGradientsKernel<<<(batch_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_dY2, d_Y, batch_size, vocab_size);

    // 2. Compute gradients for W2 and b2
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, y_desc, d_Y1, y2_desc, d_dY2, conv_desc,
                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, workSpace, workspaceSize,
                                            &beta, w2_desc, d_dW2));

    CHECK_CUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, y2_desc, d_dY2, &beta, b2_desc, d_db2));

    // 3. Compute gradients for hidden layer
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, w2_desc, d_W2, y2_desc, d_dY2, conv_desc,
                                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, workSpace, workspaceSize,
                                            &beta, y_desc, d_dY1));

    CHECK_CUDNN(cudnnActivationBackward(cudnnHandle, activationDesc, &alpha, y_desc, d_Y1, y_desc, d_dY1,
                                        y_desc, d_Y1, &beta, y_desc, d_dY1));

    // 4. Compute gradients for W1 and b1
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, x_desc, d_Xembd, y_desc, d_dY1, conv_desc,
                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, workSpace, workspaceSize,
                                            &beta, w1_desc, d_dW1));

    CHECK_CUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, y_desc, d_dY1, &beta, b1_desc, d_db1));

    // 5. Compute gradients for embeddings
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, w1_desc, d_W1, y_desc, d_dY1, conv_desc,
                                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, workSpace, workspaceSize,
                                            &beta, x_desc, d_Xembd));

    computeEmbeddingGradients<<<(batch_size * block_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_Xembd, d_X, d_dEmbd, 
                                                                batch_size, block_size, n_embed, vocab_size);

    // 6. Update weights and biases
    updateParametersKernel<<<(vocab_size * n_hidden + (threadsPerBlock - 1)) / threadsPerBlock, threadsPerBlock>>>(d_W2, d_dW2, learning_rate, vocab_size * n_hidden);
    updateParametersKernel<<<(vocab_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_b2, d_db2, learning_rate, vocab_size);
    updateParametersKernel<<<(n_hidden * block_size * n_embed + (threadsPerBlock - 1)) / threadsPerBlock, threadsPerBlock>>>(d_W1, d_dW1, learning_rate, n_hidden * block_size * n_embed);
    updateParametersKernel<<<(n_hidden + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_b1, d_db1, learning_rate, n_hidden);
    updateParametersKernel<<<(vocab_size * n_embed + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_embd, d_dEmbd, learning_rate, vocab_size * n_embed);

    // 7. Zero out the gradients
    CHECK_CUDA(cudaMemset(d_dEmbd, 0, vocab_size * n_embed * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dW1, 0, n_hidden * block_size * n_embed * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db1, 0, n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dW2, 0, vocab_size * n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_db2, 0, vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dY1, 0, batch_size * n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dY2, 0, batch_size * vocab_size * sizeof(float)));
}

__host__ size_t calculateAndAllocateWorkspace(cudnnHandle_t cudnnHandle,
                                              cudnnTensorDescriptor_t x_desc,
                                              cudnnFilterDescriptor_t w1_desc,
                                              cudnnFilterDescriptor_t w2_desc,
                                              cudnnTensorDescriptor_t y_desc,
                                              cudnnTensorDescriptor_t y2_desc,
                                              cudnnConvolutionDescriptor_t conv_desc,
                                              void **workSpace) {
    size_t workspaceSize = 0;
    size_t temp_workspaceSize = 0;
    cudnnConvolutionFwdAlgo_t fwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

    // Lambda to update workspace size
    auto updateWorkspaceSize = [&](size_t size) {
        workspaceSize = std::max(workspaceSize, size);
    };

    // Calculate workspace size for forward pass
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                x_desc,
                                                w1_desc,
                                                conv_desc,
                                                y_desc,
                                                fwdAlgo,
                                                &temp_workspaceSize));
    updateWorkspaceSize(temp_workspaceSize);

    // Calculate workspace size for backward filter (W2)
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                            y_desc,
                                                            y2_desc,
                                                            conv_desc,
                                                            w2_desc,
                                                            bwdFilterAlgo,
                                                            &temp_workspaceSize));
    updateWorkspaceSize(temp_workspaceSize);

    // Calculate workspace size for backward data (hidden layer)
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                            w2_desc,
                                                            y2_desc,
                                                            conv_desc,
                                                            y_desc,
                                                            bwdDataAlgo,
                                                            &temp_workspaceSize));
    updateWorkspaceSize(temp_workspaceSize);

    // Calculate workspace size for backward filter (W1)
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                            x_desc,
                                                            y_desc,
                                                            conv_desc,
                                                            w1_desc,
                                                            bwdFilterAlgo,
                                                            &temp_workspaceSize));
    updateWorkspaceSize(temp_workspaceSize);

    // Allocate workspace with the maximum size required
    if (workspaceSize > 0) {
        if (*workSpace) cudaFree(*workSpace);
        CHECK_CUDA(cudaMalloc(workSpace, workspaceSize));
    }

    std::cout << "Allocated workspace size: " << workspaceSize << " bytes" << std::endl;

    return workspaceSize;
}


std::vector<int> inference(
    cudnnHandle_t cudnnHandle,
    float *d_embd, float *d_W1, float *d_b1, float *d_W2, float *d_b2,
    cudnnTensorDescriptor_t x_desc, cudnnFilterDescriptor_t w1_desc,
    cudnnConvolutionDescriptor_t conv_desc, cudnnTensorDescriptor_t y_desc,
    cudnnTensorDescriptor_t b1_desc, cudnnActivationDescriptor_t activationDesc,
    cudnnFilterDescriptor_t w2_desc, cudnnTensorDescriptor_t y2_desc,
    cudnnTensorDescriptor_t b2_desc,
    void *workSpace, size_t workspaceSize,
    int *h_input, int input_length,
    int num_predictions) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate device memory for input and output
    int *d_input;
    float *d_Xembd, *d_Y1, *d_Y2;
    CHECK_CUDA(cudaMalloc(&d_input, input_length * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Xembd, input_length * n_embed * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y1, n_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y2, vocab_size * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_length * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> predictions;

    int* d_maxIndex;
    cudaMalloc(&d_maxIndex, sizeof(int));

    for (int i = 0; i < num_predictions; i++) {
        // Embedding lookup
        lookup_embeddings<<<(input_length + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_embd, d_input, d_Xembd, 1, input_length, n_embed);

        // Forward pass
        // Layer 1: Convolution + Bias + Activation
        CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, x_desc, d_Xembd, w1_desc, d_W1, conv_desc,
                                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM, workSpace, workspaceSize, &beta, y_desc, d_Y1));
        CHECK_CUDNN(cudnnAddTensor(cudnnHandle, &alpha, b1_desc, d_b1, &alpha, y_desc, d_Y1));
        CHECK_CUDNN(cudnnActivationForward(cudnnHandle, activationDesc, &alpha, y_desc, d_Y1, &beta, y_desc, d_Y1));

        // Layer 2: Convolution + Bias + Softmax
        CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, y_desc, d_Y1, w2_desc, d_W2, conv_desc,
                                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM, workSpace, workspaceSize, &beta, y2_desc, d_Y2));
        CHECK_CUDNN(cudnnAddTensor(cudnnHandle, &alpha, b2_desc, d_b2, &alpha, y2_desc, d_Y2));
        CHECK_CUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &alpha, y2_desc, d_Y2, &beta, y2_desc, d_Y2));

        // Find the token with the highest probability
        int predicted_index;
        findMaxIndex<<<1, 256>>>(d_Y2, vocab_size, d_maxIndex);
        CHECK_CUDA(cudaMemcpy(&predicted_index, d_maxIndex, sizeof(int), cudaMemcpyDeviceToHost));

        predictions.push_back(predicted_index);

        // Shift input and add the new prediction
        for (int j = 0; j < input_length - 1; j++) {
            h_input[j] = h_input[j + 1];
        }
        h_input[input_length - 1] = predicted_index;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, input_length * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_Xembd);
    cudaFree(d_Y1);
    cudaFree(d_Y2);
    cudaFree(d_maxIndex);

    return predictions;
}

int main(int argc, char *argv[]) {
    printVersions();
    // testConvolution();

    /**
    * Part 1: Prepare the training data. Extract the text from input file,
    * creat vocabulary (which is set of unique characters, also known as tokens) 
    * out of the text, 
    * Assign indexes to the vocabulary (tokens). 
    */
    std::string filename = parseCommandLineArguments(argc, argv);

    // Read and split lines from the file
    std::vector<std::string> words = readAndSplitLines(filename);
    std::cout << "Input Data size " << words.size() << std::endl;

    // Unique sorted characters and mappings
    std::set<char> charSet;
    for (const auto& w : words) {
        charSet.insert(w.begin(), w.end());
    }

    std::vector<char> chars(charSet.begin(), charSet.end());
    std::map<char, int> stoi;
    std::map<int, char> itos;
    createMappings(chars, stoi, itos);
    vocab_size = itos.size();
    std::cout << "Vocab Size is " << vocab_size << std::endl;


    // Shuffle words
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(words.begin(), words.end(), g);

    // Split the data
    size_t n1 = static_cast<size_t>(0.8 * words.size());
    size_t n2 = static_cast<size_t>(0.9 * words.size());
    std::cout << n1 << ", " << n2 << std::endl;

    auto [Xtr, Ytr, Xtr_size] = buildDataset({words.begin(), words.begin() + n1}, stoi);
    // auto [Xdev, Ydev] = buildDataset({words.begin() + n1, words.begin() + n2}, stoi);
    // auto [Xte, Yte] = buildDataset({words.begin() + n2, words.end()}, stoi);

    // Output for verification
    // std::cout << "Training set size: " << Xtr.size() << ", " << Ytr.size() << std::endl;
    // std::cout << "Validation set size: " << Xdev.size() << ", " << Ydev.size() << std::endl;
    // std::cout << "Test set size: " << Xte.size() << ", " << Yte.size() << std::endl;

   
    int *d_Xtr;
    int *d_Ytr;
    
    std::cout << "Copying input data to GPU" << std::endl;
    CHECK_CUDA(cudaMalloc(&d_Xtr, Xtr_size * block_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_Xtr, Xtr, Xtr_size * block_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_Ytr, Xtr_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_Ytr, Ytr, Xtr_size * sizeof(int), cudaMemcpyHostToDevice));

    // checkDataCopyToGPU(Xtr, Xtr_size, Ytr, d_Xtr, d_Ytr, itos);
    
    // Initialize cuDNN
    cudnnHandle_t cudnnHandle;
    CHECK_CUDNN(cudnnCreate(&cudnnHandle));

    /**
    * Part 2: Allocate all the required variables, host and device
    * memory, tensor and operation descriptors.
    */
    // Set up Array to hold indices for minibatches of training data
    int *d_idx, *d_X, *d_Y;
    float *d_Xembd, *d_embd, *d_W1, *d_b1, *d_W2, *d_b2;
    
    initialize_model_and_data(&d_idx, &d_X, &d_Y, &d_Xembd, &d_embd, 
                            &d_W1, &d_b1, &d_W2, &d_b2);
    
    // Intermediate results
    // Y1 = tanh(Xembed @ W1 + B1)
    float *d_Y1, *d_Y2, *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dY1, *d_dY2, *d_dEmbd;
    allocate_intermediate_buffers(&d_Y1, &d_Y2, &d_dW1, &d_db1, &d_dW2, &d_db2, 
                                  &d_dY1, &d_dY2, &d_dEmbd);
    

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b1_desc;
    cudnnFilterDescriptor_t w1_desc;
    cudnnTensorDescriptor_t y2_desc, b2_desc;
    cudnnFilterDescriptor_t w2_desc;
    cudnnTensorDescriptor_t labels_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activationDesc;
    initialize_cudnn_descriptors(&x_desc, &y_desc, &b1_desc, &w1_desc,
                                &y2_desc, &b2_desc, &w2_desc,
                                &labels_desc, &conv_desc, &activationDesc);

    // cross entropy loss
    float *d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));

    // Allocate workspace
    void *workSpace = nullptr;
    size_t workspaceSize = calculateAndAllocateWorkspace(cudnnHandle,
                           x_desc, w1_desc, w2_desc, y_desc, y2_desc, conv_desc,
                           &workSpace);

    // Random indices generataion setup
    curandState *devStates;
    cudaMalloc((void **)&devStates, batch_size * sizeof(curandState));
    setup_kernel<<<(batch_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(devStates, 2147483647);
    
    /**
    * Part 3: Run the training loop for the neural network.
    */
    // Training loop
    for (int i = 0; i < max_steps; ++i) {

        // Generate Random Indices
        //std::cout << "Generating random indices for training data minibatch\n";
        generate_indices_kernel<<<(batch_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_idx, devStates, batch_size, Xtr_size);
        //printRandomIndices(d_idx, itos);

        // Generate minibatch
        //std::cout << "Generating training data minibatch\n";
        generate_training_minibatch<<<(batch_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_X, d_Y, d_Xtr, d_Ytr, d_idx, batch_size, block_size);
        lookup_embeddings<<<(batch_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_embd, d_X, d_Xembd, batch_size, block_size, n_embed);
   
        
        // Forward pass
        forward_pass(cudnnHandle, d_Xembd, d_W1, d_b1, d_W2, d_b2,
                     d_Y1, d_Y2, x_desc, w1_desc, conv_desc, y_desc,
                     b1_desc, activationDesc, w2_desc, y2_desc, b2_desc,
                    workSpace, workspaceSize);

        compute_cross_entropy_loss(d_Y2, d_Y, d_loss);

        // Backpropgation
        backpropagation_and_update(cudnnHandle,
                                    d_Y2, d_dY2, d_Y1, d_dY1, d_Xembd, d_X, d_Y,
                                    d_W1, d_b1, d_W2, d_b2, d_embd,
                                    d_dW1, d_db1, d_dW2, d_db2, d_dEmbd,
                                    y2_desc, y_desc, x_desc,
                                    w2_desc, w1_desc,
                                    b2_desc, b1_desc,
                                    conv_desc,
                                    activationDesc,
                                    workSpace, workspaceSize);
       
    }

    // Run inference for a few use case
    int h_input[block_size] = {0,0,0,0,0,0,0,0,0,19,9,14,3,12,1,9};  // Your input sequence
    int num_predictions = 10;  // Number of tokens to predict

    std::vector<int> predictions = inference(cudnnHandle,
                                            d_embd, d_W1, d_b1, d_W2, d_b2,
                                            x_desc, w1_desc, conv_desc, y_desc, b1_desc, activationDesc,
                                            w2_desc, y2_desc, b2_desc,
                                            workSpace, workspaceSize, h_input, block_size, num_predictions);

    // Print predictions
    for (int token : predictions) {
        std::cout << itos.at(token) << " ";
    }
    std::cout << std::endl;


    // Clean up

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(b1_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w1_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
    
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y2_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(b2_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w2_desc));

    free(Xtr);
    free(Ytr);

    cudaFree(d_dEmbd);
    cudaFree(d_loss);
    cudaFree(devStates);
    cudaFree(d_Y1);
    cudaFree(d_Y2);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Xembd);
    cudaFree(d_idx);
    cudaFree(d_Xtr);
    cudaFree(d_Ytr);
    cudaFree(d_embd);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_dW1);
    cudaFree(d_db1);
    cudaFree(d_dW2);
    cudaFree(d_db2);
    cudaFree(d_dY1);
    cudaFree(d_dY2);
    cudnnDestroy(cudnnHandle);
    return 0;
}


// Debug functions - Ignore - not relevant to the model.
void printRandomIndices(int *d_idx, std::map<int, char>& itos) {
    int tempIdx[batch_size] = {0};
    CHECK_CUDA(cudaMemcpy(tempIdx, d_idx, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch_size; i++) {
        std::cout << tempIdx[i] << ", "; 
    }
    std::cout << std::endl;
}


void checkDataCopyToGPU(int *Xtr, int Xtr_size, int *Ytr, int *d_Xtr, int *d_Ytr, std::map<int, char>& itos){
     // Copy Input data to GPU - Since data is small this simplifiies the training loop
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < block_size; j++) {
            std::cout << itos.at(Xtr[i * block_size + j]);
        }
        std::cout << "--->" << itos.at(Ytr[i]) << std::endl;
    }

    // Read back
    int tempX[20][block_size] = {0};
    int tempY[block_size] = {0};
    CHECK_CUDA(cudaMemcpy(tempX, d_Xtr, 20 * block_size * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tempY, d_Ytr, 20 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < block_size; j++) {
            std::cout << tempX[i][j] << ",";
        }
        std::cout << "--->" << itos.at(tempY[i]) << std::endl;
    }
}


void testConvolution() {
    cudnnHandle_t cudnnHandle;
    CHECK_CUDNN(cudnnCreate(&cudnnHandle));
    
    float input[2][3] = {{1, 2, 1}, {2, 3, 1}};
    float weights[2][3] = {{1, 1, 2}, {2, 1, 3}};

    float *d_X;
    CHECK_CUDA(cudaMalloc(&d_X, 2 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, input, 2 * 3 * sizeof(float), cudaMemcpyHostToDevice));

    float *d_W;
    CHECK_CUDA(cudaMalloc(&d_W, 2 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_W, weights,  2 * 3 * sizeof(float), cudaMemcpyHostToDevice));

    float *d_Y;
    CHECK_CUDA(cudaMalloc(&d_Y,  2 * 2 * sizeof(float)));

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w1_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w1_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 2, 1, 1));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w1_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 3, 1, 1));

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    size_t workspaceSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                x_desc,
                                                w1_desc,
                                                conv_desc,
                                                y_desc,
                                                CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                                &workspaceSize));
    std::cout << "Workspace Size " << workspaceSize << std::endl;

    void* workSpace = nullptr; 
    if (workspaceSize > 0) {
        cudaMalloc(&workSpace, workspaceSize);
    }

    CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, x_desc, d_X, w1_desc, d_W, conv_desc,
                                    CUDNN_CONVOLUTION_FWD_ALGO_GEMM, workSpace, workspaceSize, &beta, y_desc, d_Y));

    float h_y[2][2] = {{0.0, 0}, {0 , 0}};
    CHECK_CUDA(cudaMemcpy(h_y, d_Y, 2 * 2  * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0 ; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << h_y[i][j] << ",";
        }
        std::cout << std::endl;
    }


    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    if (workSpace) cudaFree(workSpace);
    cudnnDestroy(cudnnHandle);
}
