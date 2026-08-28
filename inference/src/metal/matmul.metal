#include <metal_stdlib>
#include "types.metalh"
using namespace metal;

enum MatMulFlag {
    NO_TRANSPOSE = 0,
    TRANSPOSE = 1 << 0
};

// This is just dispatching GPU threads for the two outer loops that go over NxM
// So NxM operations happen in parallel instead of sequentially on CPU
kernel void matmul_naive(device const float* matrixA [[buffer(0)]],
                   constant MatMulFlag& flagsA [[buffer(1)]],
                   device const float* matrixB [[buffer(2)]],
                   constant MatMulFlag& flagsB [[buffer(3)]],
                   device float*       matrixC [[buffer(4)]],
                   constant MatrixDims& dims    [[buffer(5)]],
                   uint2 id [[thread_position_in_grid]])
{
    bool transposeA = flagsA & MatMulFlag::TRANSPOSE;
    bool transposeB = flagsB & MatMulFlag::TRANSPOSE;
    uint a_width = transposeA ? dims.M : dims.K;
    uint b_width = transposeB ? dims.K : dims.N;
    uint i = id.y;  // math: 'i' is row, so y-axis
    uint j = id.x;

    float sum = 0;
    for (uint k = 0; k < dims.K; k++) {
        uint a_row = transposeA ? k : i;
        uint a_column = transposeA ? i : k;
        uint b_row = transposeB ? j : k;
        uint b_column = transposeB ? k : j;
        sum += matrixA[a_row * a_width + a_column] * matrixB[b_row * b_width + b_column];
    }

    matrixC[i * dims.N + j] = sum;
}

#define TILE_DIM 32

// Entries are reused a lot, tiling moves memory to more local memory for faster access
kernel void matmul_tile(device const float* matrixA [[buffer(0)]],
                   constant MatMulFlag& flagsA [[buffer(1)]],
                   device const float* matrixB [[buffer(2)]],
                   constant MatMulFlag& flagsB [[buffer(3)]],
                   device float*       matrixC [[buffer(4)]],
                   constant MatrixDims& dims    [[buffer(5)]],
                   uint2 id [[thread_position_in_grid]],
                   uint2 localId [[thread_position_in_threadgroup]])
{
    threadgroup float tileA[TILE_DIM][TILE_DIM];
    threadgroup float tileB[TILE_DIM][TILE_DIM];
    bool transposeA = flagsA & MatMulFlag::TRANSPOSE;
    bool transposeB = flagsB & MatMulFlag::TRANSPOSE;

    uint n_tiles = (dims.K + TILE_DIM - 1) / TILE_DIM;

    float value = 0;
    for (uint t = 0; t < n_tiles; t++) {
        // draw a picture to understand these indices lol
        // load A into cache
        uint i_A = id.y;
        uint j_A = (t * TILE_DIM) + localId.x;
        tileA[localId.y][localId.x] = 0.0F;
        if (i_A < dims.M && j_A < dims.K) {
            if (transposeA) {
                tileA[localId.y][localId.x] = matrixA[j_A * dims.M + i_A];
            } else {
                tileA[localId.y][localId.x] = matrixA[i_A * dims.K + j_A];
            }
        }
        // load B into cache
        uint i_B = (t * TILE_DIM) + localId.y;
        uint j_B = id.x;
        tileB[localId.y][localId.x] = 0.0F;
        if (i_B < dims.K && j_B < dims.N) {
            if (transposeB) {
                tileB[localId.y][localId.x] = matrixB[j_B * dims.K + i_B];
            } else {
                tileB[localId.y][localId.x] = matrixB[i_B * dims.N + j_B];
            }
        }

        // wait for all threads to load their entry
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < TILE_DIM; k++) {
            value += tileA[localId.y][k] * tileB[k][localId.x];
        }
        // wait for all threads to finish computation before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (id.y < dims.M && id.x < dims.N) {
        matrixC[id.y * dims.N + id.x] = value;
    }
}

// For memory efficiency to load each entry exactly once, use the property of matrix multiplication where
// C[m, n] = inner_prod(mth row A, nth col B) = sum(outer_prod(kth row A, kth col B))
// so instead of load A[i, j] -> add -> release -> load... M times for each entry
// it is one load -> add for each entry
kernel void matmul_outer_prod_tile(device const float* matrixA [[buffer(0)]],
                   device const float* matrixB [[buffer(1)]],
                   device float*       matrixC [[buffer(2)]],
                   constant MatrixDims& dims    [[buffer(3)]],
                   uint2 id [[thread_position_in_grid]],
                   uint2 localId [[thread_position_in_threadgroup]])
{
    // implement this later stop sidequesting
}
