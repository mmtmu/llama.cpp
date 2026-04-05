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
    size_t old_cache_pos = 0;
    size_t len_cache = 0;
    llama_tokens cache_tokens;
    std::vector<size_t> raw_to_cache_delta;
    bool   pre_shifted = false;
};

struct server_cache_reuse_reasoning_compaction {
    size_t raw_prefix_len = 0;
    llama_tokens cache_prefix_tokens;
    std::vector<size_t> raw_to_cache_prefix;
};

std::vector<server_cache_reuse_block> server_cache_reuse_build_plan(
        const llama_tokens & tokens_old,
        const llama_tokens & tokens_new,
        size_t start_pos,
        size_t min_match,
        size_t new_limit);

std::vector<server_cache_reuse_block> server_cache_reuse_build_mapped_plan(
        const llama_tokens & tokens_old_raw,
        const llama_tokens & tokens_old_cache,
        const std::vector<size_t> & old_raw_to_cache_prefix,
        const llama_tokens & tokens_new_raw,
        size_t start_pos,
        size_t min_match,
        size_t new_limit);

bool server_cache_reuse_build_reasoning_compaction(
        const struct llama_context * ctx,
        const llama_tokens & tokens_old_raw,
        const llama_tokens & tokens_old_cache,
        const std::vector<size_t> & old_raw_to_cache_prefix,
        const llama_tokens & tokens_new_raw,
        const llama_tokens & reasoning_start,
        const llama_tokens & reasoning_end,
        server_cache_reuse_reasoning_compaction & plan);

bool server_cache_reuse_build_reasoning_compaction_pieces(
        const llama_tokens & tokens_old_raw,
        const std::vector<std::string> & pieces_old_raw,
        const llama_tokens & tokens_old_cache,
        const std::vector<std::string> & pieces_old_cache,
        const std::vector<size_t> & old_raw_to_cache_prefix,
        const llama_tokens & tokens_new_raw,
        const std::vector<std::string> & pieces_new_raw,
        const llama_tokens & reasoning_start,
        const llama_tokens & reasoning_end,
        server_cache_reuse_reasoning_compaction & plan);

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
