#include "lightning_indexer.cuh"

#include "convert.cuh"
#include "fattn-common.cuh"

constexpr int KVS_PER_WARP    = 8;
constexpr int WARPS_PER_BLOCK = 8;

template <int64_t n_embd, int64_t n_head, ggml_type type_K>
static __global__ void lightning_indexer_kernel(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23) {
    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;

    const int start_kv_block = blockIdx.x * (WARPS_PER_BLOCK * KVS_PER_WARP);
    const int start_kv       = start_kv_block + i_warp * KVS_PER_WARP;

    const char * q_base = (const char *) src0 + i_batch*nb02 + i_stream*nb03;
    const float * w_base = (const float *) ((const char *) src2 + i_batch*nb21 + i_stream*nb23);

    float4 k_vecs[KVS_PER_WARP];
    float  score_k[KVS_PER_WARP] = { 0.0f };

    constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, float, 4>();

    #pragma unroll
    for (int k = 0; k < KVS_PER_WARP; ++k) {
        const int i_kv = start_kv + k;
        if (i_kv < n_kv) {
            const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
            dequantize_k(k_base, &k_vecs[k], i_lane * 4);
        } else {
            k_vecs[k] = make_float4(0, 0, 0, 0);
        }
    }

    for (int h = 0; h < n_head; ++h) {
        const float4 q_vec = *(const float4 *) (q_base + h*nb01 + i_lane*4*sizeof(float));
        const float  w_val = w_base[h];

        float qk[KVS_PER_WARP] = { 0.0f };

        #pragma unroll
        for (int k = 0; k < KVS_PER_WARP; ++k) {
            const float4 k_vec = k_vecs[k];
            qk[k] += q_vec.x * k_vec.x;
            qk[k] += q_vec.y * k_vec.y;
            qk[k] += q_vec.z * k_vec.z;
            qk[k] += q_vec.w * k_vec.w;
        }

        #pragma unroll
        for (int k = 0; k < KVS_PER_WARP; ++k) {
            float sum = warp_reduce_sum(qk[k]);
            if (i_lane == 0) {
                sum *= scale_embd;
                sum = sum > 0.0f ? sum : 0.0f;
                score_k[k] += sum * w_val;
            }
        }
    }

    if (i_lane == 0) {
        float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
        #pragma unroll
        for (int k = 0; k < KVS_PER_WARP; ++k) {
            const int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                dst_base[i_kv] = score_k[k] * scale_heads;
            }
        }
    }
}

#define DECL_LIGHTNING_INDEXER_CASE(n_embd, n_head, type_K) \
    template __global__ void lightning_indexer_kernel<n_embd, n_head, type_K>( \
        const float * src0, const char * src1, const float * src2, float * dst, \
        const float scale_embd, const float scale_heads, \
        int64_t n_stream, int64_t n_batch, int64_t n_kv, \
        size_t nb1, size_t nb2, size_t nb3, \
        size_t nb01, size_t nb02, size_t nb03, \
        size_t nb11, size_t nb12, size_t nb13, \
        size_t nb21, size_t nb22, size_t nb23);

DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_F16)
DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_Q4_0)
DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_Q4_1)
DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_Q5_0)
DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_Q5_1)
DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_Q8_0)
DECL_LIGHTNING_INDEXER_CASE(128, 32, GGML_TYPE_BF16)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_F16)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_Q4_0)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_Q4_1)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_Q5_0)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_Q5_1)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_Q8_0)
DECL_LIGHTNING_INDEXER_CASE(128, 64, GGML_TYPE_BF16)

#define LIGHTNING_INDEXER_CASE(n_embd, n_head, K, type_K) \
    if (K->type == (type_K)) { \
        lightning_indexer_kernel<n_embd, n_head, type_K><<<grid, block, 0, ctx.stream()>>>( \
            src0_d, src1_d, src2_d, dst_d, scale_embd, scale_heads, \
            n_stream, n_batch, n_kv, \
            nb1, nb2, nb3, \
            nb01, nb02, nb03, \
            nb11, nb12, nb13, \
            nb21, nb22, nb23); \
    } else

void ggml_cuda_op_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float scale_embd  = ggml_get_op_params_f32(dst, 0);
    const float scale_heads = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb20 == ggml_type_size(src2->type));

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = src0->ne[0];
    const int n_head   = src0->ne[1];
    const int n_batch  = src0->ne[2];
    const int n_stream = src0->ne[3];
    const int n_kv     = src1->ne[2];

    const float * src0_d = (const float *) src0->data;
    const char  * src1_d = (const char  *) src1->data;
    const float * src2_d = (const float *) src2->data;
    float       * dst_d  = (float *) dst->data;

    dim3 block(32, WARPS_PER_BLOCK);
    const int num_kv_blocks = (n_kv + (KVS_PER_WARP * WARPS_PER_BLOCK) - 1) / (KVS_PER_WARP * WARPS_PER_BLOCK);
    dim3 grid(num_kv_blocks, n_batch, n_stream);

    if (n_embd == 128 && n_head == 32) {
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(128, 32, src1, GGML_TYPE_BF16)
        GGML_ABORT("fatal error");
    } else if (n_embd == 128 && n_head == 64) {
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(128, 64, src1, GGML_TYPE_BF16)
        GGML_ABORT("fatal error");
    } else {
        GGML_ABORT("unsupported lightning indexer shape: n_embd=%d n_head=%d", n_embd, n_head);
    }
}
