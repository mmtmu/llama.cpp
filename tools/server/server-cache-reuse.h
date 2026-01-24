#pragma once

#include "common.h"

#include <cstddef>
#include <vector>

struct server_cache_reuse_block {
    size_t old_pos = 0;
    size_t new_pos = 0;
    size_t len     = 0;
    bool   pre_shifted = false;
};

std::vector<server_cache_reuse_block> server_cache_reuse_build_plan(
        const llama_tokens & tokens_old,
        const llama_tokens & tokens_new,
        size_t start_pos,
        size_t min_match,
        size_t new_limit);
