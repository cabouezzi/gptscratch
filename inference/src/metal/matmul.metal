#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint M;
    uint K;
    uint N;
};

// This is just dispatching GPU threads for the two outer loops that go over NxM
// So NxM operations happen in parallel instead of sequentially on CPU
kernel void matmul(device const float* matrixA [[buffer(0)]],
                   device const float* matrixB [[buffer(1)]],
                   device float*       matrixC [[buffer(2)]],
                   constant MatrixDims& dims    [[buffer(3)]],
                   uint2 id [[thread_position_in_grid]])
{
    uint i = id.y;  // math: 'i' is row, so y-axis
    uint j = id.x;

    float sum = 0;
    for (uint k = 0; k < dims.K; k++) {
        sum += matrixA[i * dims.K + k] * matrixB[k * dims.N + j];
    }

    matrixC[i * dims.N + j] = sum;
}