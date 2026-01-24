#include "arg.h"
#include "common.h"
#include "get-model.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static bool run_case(llama_context * ctx, llama_seq_id seq_id, int n_prefix, int old_pos, int new_pos, int len) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        fprintf(stderr, "%s : no memory available\n", __func__);
        return false;
    }

    llama_memory_seq_rm(mem, seq_id, -1, -1);

    const int total_tokens = std::max(old_pos + len + 8, new_pos + len + 8);
    std::vector<llama_token> tokens(total_tokens, 1);

    llama_batch batch = llama_batch_init(total_tokens, 0, 1);
    for (int i = 0; i < total_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode initial prompt\n", __func__);
        llama_batch_free(batch);
        return false;
    }
    llama_batch_free(batch);

    const llama_pos p0 = (llama_pos) old_pos;
    const llama_pos p1 = (llama_pos) (old_pos + len);

    const size_t stash_size = llama_memory_seq_get_size_range(mem, seq_id, p0, p1);
    if (stash_size == 0) {
        fprintf(stderr, "%s : failed to get stash size\n", __func__);
        return false;
    }

    std::vector<uint8_t> stash(stash_size);
    const size_t stash_written = llama_memory_seq_get_data_range(mem, stash.data(), stash.size(), seq_id, p0, p1);
    if (stash_written != stash.size()) {
        fprintf(stderr, "%s : failed to stash range data\n", __func__);
        return false;
    }

    if (!llama_memory_seq_rm(mem, seq_id, (llama_pos) n_prefix, -1)) {
        fprintf(stderr, "%s : failed to clear range\n", __func__);
        return false;
    }

    if (new_pos < n_prefix) {
        fprintf(stderr, "%s : new_pos < n_prefix\n", __func__);
        return false;
    }

    const int gap = new_pos - n_prefix;
    if (gap > 0) {
        llama_batch gap_batch = llama_batch_init(gap, 0, 1);
        for (int i = 0; i < gap; i++) {
            common_batch_add(gap_batch, tokens[0], n_prefix + i, { seq_id }, false);
        }
        gap_batch.logits[gap_batch.n_tokens - 1] = true;
        if (llama_decode(ctx, gap_batch)) {
            fprintf(stderr, "%s : failed to decode gap tokens\n", __func__);
            llama_batch_free(gap_batch);
            return false;
        }
        llama_batch_free(gap_batch);
    }

    const llama_pos pos_shift = (llama_pos) (new_pos - old_pos);
    const size_t stash_read = llama_memory_seq_set_data_range(mem, stash.data(), stash.size(), seq_id, pos_shift);
    if (stash_read != stash.size()) {
        fprintf(stderr, "%s : failed to restore range data\n", __func__);
        return false;
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);
    const llama_pos expected_max = (llama_pos) (new_pos + len - 1);
    if (pos_max != expected_max) {
        fprintf(stderr, "%s : unexpected pos_max %d (expected %d)\n", __func__, pos_max, expected_max);
        return false;
    }

    llama_batch next_batch = llama_batch_init(1, 0, 1);
    common_batch_add(next_batch, tokens[0], new_pos + len, { seq_id }, true);
    if (llama_decode(ctx, next_batch)) {
        fprintf(stderr, "%s : failed to decode after restore\n", __func__);
        llama_batch_free(next_batch);
        return false;
    }
    llama_batch_free(next_batch);

    return true;
}

int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    common_params params;
    params.model.path = model_path;
    params.n_ctx = 256;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_parallel = 1;
    params.n_sequences = 1;

    common_init();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();

    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return 1;
    }

    if (!run_case(ctx, 0, 4, 8, 12, 6)) {
        fprintf(stderr, "%s : FAILED\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : SUCCESS\n", __func__);
    return 0;
}
