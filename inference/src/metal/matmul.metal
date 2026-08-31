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
#define THREAD_TILE_ROWS 2
#define THREAD_TILE_COLUMNS 4
#define THREADGROUP_ROWS (TILE_DIM / THREAD_TILE_ROWS)
#define THREADGROUP_COLUMNS (TILE_DIM / THREAD_TILE_COLUMNS)

// Entries are reused a lot, tiling moves memory to more local memory for faster access
kernel void matmul_tile(device const float* matrixA [[buffer(0)]],
                   constant MatMulFlag& flagsA [[buffer(1)]],
                   device const float* matrixB [[buffer(2)]],
                   constant MatMulFlag& flagsB [[buffer(3)]],
                   device float*       matrixC [[buffer(4)]],
                   constant MatrixDims& dims    [[buffer(5)]],
                   uint2 threadgroupId [[threadgroup_position_in_grid]],
                   uint2 localId [[thread_position_in_threadgroup]])
{
    threadgroup float tileA[TILE_DIM][TILE_DIM];
    threadgroup float4 tileB[TILE_DIM][TILE_DIM / 4];
    bool transposeA = flagsA & MatMulFlag::TRANSPOSE;
    bool transposeB = flagsB & MatMulFlag::TRANSPOSE;

    uint n_tiles = (dims.K + TILE_DIM - 1) / TILE_DIM;
    uint outputTileRow = threadgroupId.y * TILE_DIM;
    uint outputTileColumn = threadgroupId.x * TILE_DIM;
    uint outputRow0 = outputTileRow + localId.y;
    uint outputRow1 = outputRow0 + THREADGROUP_ROWS;
    uint outputColumn = outputTileColumn + localId.x * THREAD_TILE_COLUMNS;

    float4 outputRowVector0 = float4(0.0F);
    float4 outputRowVector1 = float4(0.0F);
    for (uint t = 0; t < n_tiles; t++) {
        uint localThreadIndex = localId.y * THREADGROUP_COLUMNS + localId.x;
        uint threadgroupSize = THREADGROUP_ROWS * THREADGROUP_COLUMNS;
        uint tileElementCount = TILE_DIM * TILE_DIM;

        for (uint index = localThreadIndex; index < tileElementCount;
             index += threadgroupSize) {
            uint tileRow = index / TILE_DIM;
            uint tileColumn = index % TILE_DIM;

            uint aRow = outputTileRow + tileRow;
            uint aColumn = t * TILE_DIM + tileColumn;
            tileA[tileRow][tileColumn] = 0.0F;
            if (aRow < dims.M && aColumn < dims.K) {
                if (transposeA) {
                    tileA[tileRow][tileColumn] = matrixA[aColumn * dims.M + aRow];
                } else {
                    tileA[tileRow][tileColumn] = matrixA[aRow * dims.K + aColumn];
                }
            }
        }

        uint tileVectorCount = TILE_DIM * (TILE_DIM / 4);
        for (uint index = localThreadIndex; index < tileVectorCount;
             index += threadgroupSize) {
            uint tileRow = index / (TILE_DIM / 4);
            uint tileVectorColumn = index % (TILE_DIM / 4);
            uint bRow = t * TILE_DIM + tileRow;
            uint bColumn = outputTileColumn + tileVectorColumn * 4;
            float4 b = float4(0.0F);

            for (uint component = 0; component < 4; component++) {
                uint column = bColumn + component;
                if (bRow < dims.K && column < dims.N) {
                    if (transposeB) {
                        b[component] = matrixB[column * dims.K + bRow];
                    } else {
                        b[component] = matrixB[bRow * dims.N + column];
                    }
                }
            }

            tileB[tileRow][tileVectorColumn] = b;
        }

        // wait for all threads to load their entry
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_DIM; k++) {
            float a0 = tileA[localId.y][k];
            float a1 = tileA[localId.y + THREADGROUP_ROWS][k];
            float4 b = tileB[k][localId.x];
            outputRowVector0 += a0 * b;
            outputRowVector1 += a1 * b;
        }
        // wait for all threads to finish computation before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint columnOffset = 0; columnOffset < THREAD_TILE_COLUMNS;
         columnOffset++) {
        uint column = outputColumn + columnOffset;
        if (column < dims.N) {
            if (outputRow0 < dims.M) {
                matrixC[outputRow0 * dims.N + column] = outputRowVector0[columnOffset];
            }
            if (outputRow1 < dims.M) {
                matrixC[outputRow1 * dims.N + column] = outputRowVector1[columnOffset];
            }
        }
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
