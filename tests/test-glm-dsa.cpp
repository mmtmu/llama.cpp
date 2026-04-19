#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

static ggml_context_ptr make_ctx(size_t size = 2u * 1024u * 1024u) {
    ggml_init_params params = {
        /*.mem_size   = */ size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static std::vector<float> read_tensor_f32(const ggml_tensor * tensor) {
    std::vector<float> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

static std::vector<int32_t> read_tensor_i32(const ggml_tensor * tensor) {
    std::vector<int32_t> out(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data(), 0, out.size() * sizeof(int32_t));
    return out;
}

static void write_tensor_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    GGML_ASSERT((int64_t) values.size() == ggml_nelements(tensor));
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static void write_tensor_f16(ggml_tensor * tensor, const std::vector<float> & values) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F16);
    GGML_ASSERT((int64_t) values.size() == ggml_nelements(tensor));

    std::vector<ggml_fp16_t> values_f16(values.size());
    ggml_fp32_to_fp16_row(values.data(), values_f16.data(), values.size());
    ggml_backend_tensor_set(tensor, values_f16.data(), 0, values_f16.size() * sizeof(ggml_fp16_t));
}

static ggml_backend_buffer_ptr allocate_ctx_tensors(ggml_backend_t backend, ggml_context * ctx) {
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx, backend));
    GGML_ASSERT(buf != nullptr);
    return buf;
}

static void compute_graph(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
}

static std::vector<float> reference_lightning_indexer(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & weights,
        int64_t n_embd,
        int64_t n_head,
        int64_t n_batch,
        int64_t n_kv,
        int64_t n_stream,
        float scale_embd,
        float scale_heads) {
    std::vector<float> out(n_kv * n_batch * n_stream, 0.0f);

    auto q_idx = [=](int64_t embd, int64_t head, int64_t batch, int64_t stream) {
        return embd + n_embd * (head + n_head * (batch + n_batch * stream));
    };
    auto k_idx = [=](int64_t embd, int64_t kv, int64_t stream) {
        return embd + n_embd * (kv + n_kv * stream);
    };
    auto w_idx = [=](int64_t head, int64_t batch, int64_t stream) {
        return head + n_head * (batch + n_batch * stream);
    };
    auto out_idx = [=](int64_t kv, int64_t batch, int64_t stream) {
        return kv + n_kv * (batch + n_batch * stream);
    };

    for (int64_t stream = 0; stream < n_stream; ++stream) {
        for (int64_t batch = 0; batch < n_batch; ++batch) {
            for (int64_t kv = 0; kv < n_kv; ++kv) {
                float score = 0.0f;
                for (int64_t head = 0; head < n_head; ++head) {
                    float qk = 0.0f;
                    for (int64_t embd = 0; embd < n_embd; ++embd) {
                        qk += q[q_idx(embd, head, batch, stream)] * k[k_idx(embd, kv, stream)];
                    }
                    score += std::max(qk * scale_embd, 0.0f) * weights[w_idx(head, batch, stream)];
                }
                out[out_idx(kv, batch, stream)] = score * scale_heads;
            }
        }
    }

    return out;
}

static bool all_close(
        testing & t,
        const std::vector<float> & got,
        const std::vector<float> & expected,
        float atol,
        const std::string & label) {
    if (!t.assert_true(label + ": size mismatch", got.size() == expected.size())) {
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > atol) {
            ok = false;
            t.assert_true(
                    label + ": mismatch at " + std::to_string(i) +
                    ", got=" + std::to_string(got[i]) +
                    ", expected=" + std::to_string(expected[i]),
                    false);
            break;
        }
    }

    if (ok) {
        t.assert_true(true);
    }
    return ok;
}

static void test_lightning_indexer_reference(testing & t, ggml_type type_k) {
    const int64_t n_embd   = 4;
    const int64_t n_head   = 3;
    const int64_t n_batch  = 2;
    const int64_t n_kv     = 5;
    const int64_t n_stream = 1;
    const float scale_embd  = 0.25f;
    const float scale_heads = 0.5f;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto ctx = make_ctx();

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_embd, n_head, n_batch, n_stream);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), type_k,        n_embd, 1,      n_kv,    n_stream);
    ggml_tensor * w = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_head, n_batch, 1,      n_stream);
    ggml_tensor * out = ggml_lightning_indexer(ctx.get(), q, k, w, scale_embd, scale_heads);
    auto buf = allocate_ctx_tensors(backend.get(), ctx.get());

    const std::vector<float> q_data = {
        0.50f, -1.00f, 0.25f,  1.50f,
       -0.25f,  0.75f, 1.25f, -0.50f,
        1.00f,  0.50f, -0.75f, 0.00f,
        0.20f, -0.40f, 0.60f,  0.80f,
       -1.20f,  0.30f, 0.90f, -0.70f,
        0.10f,  1.10f, -0.20f, 0.40f,
    };
    const std::vector<float> k_data = {
        0.30f, -0.60f, 0.90f, -1.20f,
        0.80f,  0.50f, -0.40f, 0.70f,
       -0.10f,  0.20f, 0.30f,  0.40f,
        1.00f, -0.50f, 0.25f, -0.75f,
       -0.90f,  0.60f, 0.10f,  0.20f,
    };
    const std::vector<float> w_data = {
        0.50f, -0.25f, 1.00f,
        1.50f,  0.75f, -0.50f,
    };

    write_tensor_f32(q, q_data);
    write_tensor_f32(w, w_data);
    if (type_k == GGML_TYPE_F16) {
        write_tensor_f16(k, k_data);
    } else {
        write_tensor_f32(k, k_data);
    }

    compute_graph(backend.get(), ctx.get(), out);

    const std::vector<float> got = read_tensor_f32(out);
    const std::vector<float> expected = reference_lightning_indexer(
            q_data, k_data, w_data, n_embd, n_head, n_batch, n_kv, n_stream, scale_embd, scale_heads);

    all_close(t, got, expected, type_k == GGML_TYPE_F16 ? 1e-3f : 1e-6f, "lightning_indexer");
}

static std::vector<float> reference_sparse_mask(
        const std::vector<float> & base_mask,
        const std::vector<int32_t> & top_k,
        int64_t n_kv,
        int64_t n_tokens,
        int64_t n_top_k) {
    std::vector<float> out = base_mask;

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t i = 0; i < n_top_k; ++i) {
            const int32_t kv = top_k[i + n_top_k * token];
            out[kv + n_kv * token] = base_mask[kv + n_kv * token];
        }

        for (int64_t kv = 0; kv < n_kv; ++kv) {
            bool selected = false;
            for (int64_t i = 0; i < n_top_k; ++i) {
                if (top_k[i + n_top_k * token] == kv) {
                    selected = true;
                    break;
                }
            }
            if (!selected) {
                out[kv + n_kv * token] = -std::numeric_limits<float>::infinity() + base_mask[kv + n_kv * token];
            }
        }
    }

    return out;
}

static void test_sparse_mask_construction(testing & t) {
    const int64_t n_kv = 6;
    const int64_t n_tokens = 3;
    const int64_t n_top_k = 2;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto ctx = make_ctx();

    ggml_tensor * score   = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_kv, n_tokens, 1, 1);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, n_kv, n_tokens, 1, 1);

    ggml_tensor * top_k = ggml_top_k(ctx.get(), score, n_top_k);
    ggml_tensor * kq_mask_all = ggml_fill(ctx.get(), kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx.get(), kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    top_k = ggml_view_4d(ctx.get(), top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3] * top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, top_k->ne[0], top_k->ne[1], top_k->ne[2]);
    zeros = ggml_fill(ctx.get(), zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx.get(), kq_mask_all, zeros, top_k);
    kq_mask_top_k = ggml_view_4d(ctx.get(), kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);
    ggml_tensor * sparse_mask = ggml_add(ctx.get(), kq_mask_top_k, kq_mask);
    auto buf = allocate_ctx_tensors(backend.get(), ctx.get());

    const std::vector<float> score_data = {
        0.1f, 0.6f, 0.2f, 0.9f, 0.4f, 0.3f,
        0.8f, 0.2f, 0.5f, 0.1f, 0.7f, 0.4f,
        0.3f, 0.4f, 0.9f, 0.2f, 0.8f, 0.1f,
    };
    const std::vector<float> mask_data = {
        0.0f, -100.0f, -1000.0f, -1000.0f, -1000.0f, -1000.0f,
        0.0f,    0.0f, -100.0f, -1000.0f, -1000.0f, -1000.0f,
        0.0f,    0.0f,    0.0f, -100.0f,  -1000.0f, -1000.0f,
    };

    write_tensor_f32(score, score_data);
    write_tensor_f32(kq_mask, mask_data);

    compute_graph(backend.get(), ctx.get(), sparse_mask);

    const std::vector<int32_t> got_top_k = read_tensor_i32(top_k);
    const std::vector<float> got_mask = read_tensor_f32(sparse_mask);
    const std::vector<float> expected_mask = reference_sparse_mask(mask_data, got_top_k, n_kv, n_tokens, n_top_k);

    all_close(t, got_mask, expected_mask, 0.0f, "sparse_mask");
}

#ifdef GGML_USE_CUDA
static void test_cuda_support_predicate(testing & t) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        return;
    }

    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    auto ctx = make_ctx();

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 128, 32, 1, 1);
    ggml_tensor * k_f16 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 128, 1, 16, 1);
    ggml_tensor * k_f32 = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 128, 1, 16, 1);
    ggml_tensor * w = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 32, 1, 1, 1);

    ggml_tensor * op_f16 = ggml_lightning_indexer(ctx.get(), q, k_f16, w, 1.0f, 1.0f);
    ggml_tensor * op_f32 = ggml_lightning_indexer(ctx.get(), q, k_f32, w, 1.0f, 1.0f);

    t.assert_true("cuda should support lightning_indexer with f16 K", ggml_backend_supports_op(backend.get(), op_f16));
    t.assert_true("cuda should reject lightning_indexer with f32 K", !ggml_backend_supports_op(backend.get(), op_f32));
}
#endif

int main() {
    testing t;

    t.test("lightning_indexer_f32_k_reference", [](testing & t) {
        test_lightning_indexer_reference(t, GGML_TYPE_F32);
    });

    t.test("lightning_indexer_f16_k_reference", [](testing & t) {
        test_lightning_indexer_reference(t, GGML_TYPE_F16);
    });

    t.test("sparse_mask_construction", [](testing & t) {
        test_sparse_mask_construction(t);
    });

#ifdef GGML_USE_CUDA
    t.test("cuda_support_predicate", [](testing & t) {
        test_cuda_support_predicate(t);
    });
#endif

    return t.failures == 0 ? 0 : 1;
}
