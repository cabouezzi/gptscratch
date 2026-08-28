#include <metal_stdlib>
#include "types.metalh"
using namespace metal;

inline void load_tile_to_threadgroup(
    device const float4*  src_global,
    threadgroup float4*   dst_shared, 
    uint                  laneId,
    uint                  total_float4s
) {
    for (uint i = laneId; i < total_float4s; i += 32 /* threadgroup size */) {
        dst_shared[i] = src_global[i];
    }
}

// Is this the final boss 
kernel void scaled_dot_product_attention(device const float* Q [[buffer(0)]],
                   device const float* K [[buffer(1)]],
                   device const float* V [[buffer(2)]],
                   device float*      O [[buffer(3)]],
                   constant MatrixDims& dims    [[buffer(4)]],
                   constant bool& isCausal [[buffer(5)]],
                   // each threadblock is one row of Q
                   // updated to 2D to handle multiple attention heads
                   uint2 threadgroupId [[threadgroup_position_in_grid]],
                   uint laneId [[thread_index_in_simdgroup]])
{
    uint rowId = threadgroupId.x;
    uint headId = threadgroupId.y;
    constexpr uint TILE_K = 64;
    // 64 / 4. tryna see how can i make this a variable to pass so im not constrained here
    constexpr uint VECTORS_PER_ROW = 16;
    const uint d = dims.K;
    const uint head_offset = dims.N * d;
    const bool active = laneId < VECTORS_PER_ROW;

    device const float4* row_ptr = reinterpret_cast<device const float4*>(
        Q + (headId * head_offset) + (rowId * d));
    float4 q_register = active ? row_ptr[laneId] : float4(0.0f);

    threadgroup float4 tileK[TILE_K][VECTORS_PER_ROW];
    threadgroup float4 tileV[TILE_K][VECTORS_PER_ROW];
    const uint TILE_FLOAT4S = TILE_K * (d / 4);
    const uint NUM_KV_TILES = dims.N / TILE_K;

    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float4 o_acc = float4(0.0f);

    for (uint kv_tile = 0; kv_tile < NUM_KV_TILES; kv_tile++) {
        uint tile_offset = kv_tile * TILE_FLOAT4S;
        device const float4* K4 = reinterpret_cast<device const float4*>(K + (headId * head_offset));
        device const float4* V4 = reinterpret_cast<device const float4*>(V + (headId * head_offset));
        load_tile_to_threadgroup(K4 + tile_offset, &tileK[0][0], laneId, TILE_FLOAT4S);
        load_tile_to_threadgroup(V4 + tile_offset, &tileV[0][0], laneId, TILE_FLOAT4S);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // calculate Q_i K_i^T
        // Store scores for K tile in local registers
        float scores[TILE_K];
        float scale = 1.0f / sqrt(float(d));
        // j cuz transpose
        for (uint j = 0; j < TILE_K; j++) {
            // j is the row, 32 float4's per row, and laneId is the "row tile" we're indexing into
            float4 k_reg = active ? tileK[j][laneId] : float4(0.0f);
            // had no idea dot was a function shoutout the tutorial
            float partial_dot = dot(q_register, k_reg);
            float score = simd_sum(partial_dot);  // look into simd in the GPU its different from CPU o.O
            uint keyRow = (kv_tile * TILE_K) + j;
            bool visible = !isCausal || keyRow <= rowId;
            scores[j] = visible ? score * scale : -INFINITY;
        }

        // get max for the tile
        float m_curr = -INFINITY;
        for (uint j = 0; j < TILE_K; j++) {
            m_curr = max(m_curr, scores[j]);
        }
        // calculate adjustment factors for accumulator
        float m_new = max(m_prev, m_curr);
        float alpha = exp(m_prev - m_new);
        // adjust for future calculation
        float d_curr = 0.0f;
        for (uint j = 0; j < TILE_K; j++) {
            scores[j] = exp(scores[j] - m_new);
            d_curr += scores[j];
        }
        // adjust existing accumulator
        o_acc *= alpha;

        // multiply by V post softmax
        if (active) {
            for (uint j = 0; j < TILE_K; j++) {
                float4 v_reg = tileV[j][laneId];
                o_acc += scores[j] * v_reg;
            }
        }

        d_prev = (d_prev * alpha) + d_curr;
        m_prev = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        o_acc /= d_prev;
        device float4* out_ptr = reinterpret_cast<device float4*>(O + (headId * head_offset) + (rowId * d));
        out_ptr[laneId] = o_acc;
    }
}
