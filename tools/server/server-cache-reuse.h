#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <limits>
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

struct server_cache_reuse_routing_candidate {
    const llama_tokens * tokens_old = nullptr;
    int64_t t_last_used = 0;
    int32_t slot_id = -1;
};

struct server_cache_reuse_routing_choice {
    size_t candidate_index = std::numeric_limits<size_t>::max();
    size_t sum_len = 0;
    size_t max_len = 0;
};

server_cache_reuse_routing_choice server_cache_reuse_select_slot(
        const std::vector<server_cache_reuse_routing_candidate> & candidates,
        const llama_tokens & tokens_new,
        size_t min_match);
