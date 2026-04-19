#include "models.h"

#include "llama-kv-cache-dsa.h"

llm_build_glm_dsa::llm_build_glm_dsa(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.is_mla());

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_attn_k_dsa * inp_attn_dsa = build_attn_inp_k_dsa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int effective_n_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < effective_n_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * qr = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
        cb(qr, "qr", il);

        qr = build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "qr", il);

        ggml_tensor * top_k = nullptr;

        {
            ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
            cb(indexer_q, "indexer_q", il);

            ggml_tensor * indexer_q_nope = ggml_view_3d(
                    ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                    ggml_row_size(indexer_q->type, n_embd_indexer_head),
                    ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                    0);
            ggml_tensor * indexer_q_pe = ggml_view_3d(
                    ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                    ggml_row_size(indexer_q->type, n_embd_indexer_head),
                    ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                    ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

            indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(indexer_q_pe, "indexer_q_pe", il);

            indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
            cb(indexer_q, "indexer_q", il);

            ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
            cb(indexer_k, "indexer_k", il);

            indexer_k = build_norm(indexer_k, model.layers[il].indexer_k_norm, model.layers[il].indexer_k_norm_b, LLM_NORM, il);
            cb(indexer_k, "indexer_k", il);

            ggml_tensor * indexer_k_nope = ggml_view_3d(
                    ctx0, indexer_k, n_embd_indexer_head_nope, 1, n_tokens,
                    ggml_row_size(indexer_k->type, n_embd_indexer_head),
                    ggml_row_size(indexer_k->type, n_embd_indexer_head),
                    0);
            ggml_tensor * indexer_k_pe = ggml_view_3d(
                    ctx0, indexer_k, n_embd_indexer_head_rope, 1, n_tokens,
                    ggml_row_size(indexer_k->type, n_embd_indexer_head),
                    ggml_row_size(indexer_k->type, n_embd_indexer_head),
                    ggml_row_size(indexer_k->type, n_embd_indexer_head_nope));

            indexer_k_pe = ggml_rope_ext(ctx0, indexer_k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(indexer_k_pe, "indexer_k_pe", il);

            indexer_k = ggml_concat(ctx0, indexer_k_nope, indexer_k_pe, 0);
            cb(indexer_k, "indexer_k", il);

            indexer_q = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q);
            indexer_k = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_k);

            const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
            const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
            ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

            ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
            cb(indexer_weights, "indexer_weights", il);

            indexer_k = mctx_lid->get_k(ctx0, il);

            const auto n_stream = indexer_k->ne[3];
            indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
                    indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
            indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
                    indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

            ggml_tensor * indexer_score = ggml_lightning_indexer(
                    ctx0, indexer_q, indexer_k, indexer_weights,
                    1.0f / sqrtf(float(n_embd_indexer_head)),
                    1.0f / sqrtf(float(n_indexer_head)));
            cb(indexer_score, "indexer_score", il);

            indexer_score = ggml_add(ctx0, indexer_score, inp_attn_dsa->get_kq_mask_lid());
            cb(indexer_score, "indexer_score_masked", il);

            const uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k ? indexer_score->ne[0] : n_indexer_top_k;
            top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
            cb(top_k, "top_k", il);
        }

        ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_b, qr);
        cb(q, "q", il);

        ggml_tensor * q_nope = ggml_view_3d(
                ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head,
                0);
        ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
        cb(kv_cmpr_pe, "kv_cmpr_pe", il);

        ggml_tensor * kv_cmpr = ggml_view_2d(
                ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        ggml_tensor * k_pe = ggml_view_3d(
                ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

        kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(kv_cmpr, "kv_cmpr", il);

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        ggml_tensor * Vcur = kv_cmpr;

        cur = build_attn(inp_attn_dsa,
                model.layers[il].wo, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);

        if (il == effective_n_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
