// Vector Addition Naive

__global__ void vector_add(const float *A,
                         const float *B,
                         float *C, 
                         size_t N){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}
