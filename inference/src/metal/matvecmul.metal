#include <metal_stdlib>
using namespace metal;

constant uint SIMD_WIDTH = 32;

struct HeadProjectionOffsets {
    ulong queryWeight;
    ulong keyWeight;
    ulong valueWeight;
    ulong queryOutput;
    ulong keyOutput;
    ulong valueOutput;
};

kernel void matvecmul(
    device const float4* input [[buffer(0)]],
    device const float4* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& inputSize [[buffer(3)]],
    constant uint& outputSize [[buffer(4)]],
    uint threadgroupId [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]])
{
    uint inputVectorCount = inputSize / 4;
    uint outputStart = threadgroupId * 4;
    float4 partialSums = float4(0.0F);

    for (uint inputVector = laneId; inputVector < inputVectorCount;
         inputVector += SIMD_WIDTH) {
        float4 inputValues = input[inputVector];

        if (outputStart < outputSize) {
            partialSums.x += dot(
                inputValues,
                weights[outputStart * inputVectorCount + inputVector]);
        }
        if (outputStart + 1 < outputSize) {
            partialSums.y += dot(
                inputValues,
                weights[(outputStart + 1) * inputVectorCount + inputVector]);
        }
        if (outputStart + 2 < outputSize) {
            partialSums.z += dot(
                inputValues,
                weights[(outputStart + 2) * inputVectorCount + inputVector]);
        }
        if (outputStart + 3 < outputSize) {
            partialSums.w += dot(
                inputValues,
                weights[(outputStart + 3) * inputVectorCount + inputVector]);
        }
    }

    float4 sums = float4(
        simd_sum(partialSums.x),
        simd_sum(partialSums.y),
        simd_sum(partialSums.z),
        simd_sum(partialSums.w));

    if (laneId == 0) {
        for (uint outputOffset = 0; outputOffset < 4; outputOffset++) {
            uint outputIndex = outputStart + outputOffset;
            if (outputIndex < outputSize) {
                output[outputIndex] = sums[outputOffset];
            }
        }
    }
}

kernel void qkv_matvecmul_heads(
    device const float4* input [[buffer(0)]],
    device const float4* weights [[buffer(1)]],
    device float* queryOutput [[buffer(2)]],
    device float* keyOutput [[buffer(3)]],
    device float* valueOutput [[buffer(4)]],
    constant HeadProjectionOffsets* offsets [[buffer(5)]],
    constant uint& inputSize [[buffer(6)]],
    constant uint& headSize [[buffer(7)]],
    constant uint& headCount [[buffer(8)]],
    uint2 threadgroupId [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]])
{
    uint head = threadgroupId.y;
    if (head >= headCount) {
        return;
    }

    uint inputVectorCount = inputSize / 4;
    uint outputStart = threadgroupId.x * 4;
    HeadProjectionOffsets headOffsets = offsets[head];
    float4 queryPartialSums = float4(0.0F);
    float4 keyPartialSums = float4(0.0F);
    float4 valuePartialSums = float4(0.0F);

    for (uint inputVector = laneId; inputVector < inputVectorCount;
         inputVector += SIMD_WIDTH) {
        float4 inputValues = input[inputVector];
        for (uint outputOffset = 0; outputOffset < 4; outputOffset++) {
            uint outputIndex = outputStart + outputOffset;
            if (outputIndex < headSize) {
                ulong weightIndex =
                    ulong(outputIndex) * inputVectorCount + inputVector;
                queryPartialSums[outputOffset] += dot(
                    inputValues,
                    weights[headOffsets.queryWeight + weightIndex]);
                keyPartialSums[outputOffset] += dot(
                    inputValues,
                    weights[headOffsets.keyWeight + weightIndex]);
                valuePartialSums[outputOffset] += dot(
                    inputValues,
                    weights[headOffsets.valueWeight + weightIndex]);
            }
        }
    }

    float4 querySums = float4(
        simd_sum(queryPartialSums.x),
        simd_sum(queryPartialSums.y),
        simd_sum(queryPartialSums.z),
        simd_sum(queryPartialSums.w));
    float4 keySums = float4(
        simd_sum(keyPartialSums.x),
        simd_sum(keyPartialSums.y),
        simd_sum(keyPartialSums.z),
        simd_sum(keyPartialSums.w));
    float4 valueSums = float4(
        simd_sum(valuePartialSums.x),
        simd_sum(valuePartialSums.y),
        simd_sum(valuePartialSums.z),
        simd_sum(valuePartialSums.w));

    if (laneId == 0) {
        for (uint outputOffset = 0; outputOffset < 4; outputOffset++) {
            uint outputIndex = outputStart + outputOffset;
            if (outputIndex < headSize) {
                queryOutput[headOffsets.queryOutput + outputIndex] =
                    querySums[outputOffset];
                keyOutput[headOffsets.keyOutput + outputIndex] =
                    keySums[outputOffset];
                valueOutput[headOffsets.valueOutput + outputIndex] =
                    valueSums[outputOffset];
            }
        }
    }
}
