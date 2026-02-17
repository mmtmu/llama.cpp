#include "server-cache-reuse.h"

#include <algorithm>
#include <unordered_map>

static uint64_t server_cache_reuse_token_value(const llama_token token) {
    return (uint64_t) (uint32_t) token + 1ull;
}

static size_t server_cache_reuse_find_first_ge(const std::vector<size_t> & values, const size_t target) {
    size_t lo = 0;
    size_t hi = values.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (values[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

std::vector<server_cache_reuse_block> server_cache_reuse_build_plan(
        const llama_tokens & tokens_old,
        const llama_tokens & tokens_new,
        size_t start_pos,
        size_t min_match,
        size_t new_limit) {
    std::vector<server_cache_reuse_block> plan;

    if (min_match == 0) {
        return plan;
    }

    const size_t size_old = tokens_old.size();
    const size_t size_new = tokens_new.size();
    if (start_pos >= size_old || start_pos >= size_new) {
        return plan;
    }

    if (new_limit == 0 || new_limit > size_new) {
        new_limit = size_new;
    }

    if (start_pos + min_match > size_old || start_pos + min_match > new_limit) {
        return plan;
    }

    const uint64_t base = 0x9e3779b97f4a7c15ull;
    uint64_t base_pow = 1;
    for (size_t i = 0; i < min_match; i++) {
        base_pow *= base;
    }

    std::unordered_map<uint64_t, std::vector<size_t>> old_windows;
    old_windows.reserve(size_old - start_pos);

    uint64_t hash_old = 0;
    for (size_t i = 0; i < min_match; i++) {
        hash_old = hash_old * base + server_cache_reuse_token_value(tokens_old[start_pos + i]);
    }

    for (size_t pos = start_pos; pos + min_match <= size_old; pos++) {
        old_windows[hash_old].push_back(pos);

        if (pos + min_match >= size_old) {
            break;
        }

        const uint64_t out_val = server_cache_reuse_token_value(tokens_old[pos]);
        const uint64_t in_val  = server_cache_reuse_token_value(tokens_old[pos + min_match]);
        hash_old = hash_old * base + in_val;
        hash_old -= out_val * base_pow;
    }

    size_t new_pos = start_pos;
    size_t old_min = start_pos;

    while (new_pos + min_match <= new_limit) {
        uint64_t hash_new = 0;
        for (size_t i = 0; i < min_match; i++) {
            hash_new = hash_new * base + server_cache_reuse_token_value(tokens_new[new_pos + i]);
        }

        bool matched = false;
        auto it = old_windows.find(hash_new);
        if (it != old_windows.end()) {
            const std::vector<size_t> & candidates = it->second;
            size_t idx = server_cache_reuse_find_first_ge(candidates, old_min);
            for (; idx < candidates.size(); idx++) {
                const size_t old_pos = candidates[idx];
                if (old_pos + min_match > size_old) {
                    break;
                }

                bool ok = true;
                for (size_t i = 0; i < min_match; i++) {
                    if (tokens_old[old_pos + i] != tokens_new[new_pos + i]) {
                        ok = false;
                        break;
                    }
                }

                if (!ok) {
                    continue;
                }

                size_t len = min_match;
                while (old_pos + len < size_old &&
                       new_pos + len < new_limit &&
                       tokens_old[old_pos + len] == tokens_new[new_pos + len]) {
                    len++;
                }

                plan.push_back(server_cache_reuse_block{ old_pos, new_pos, len, false });
                old_min = old_pos + len;
                new_pos += len;
                matched = true;
                break;
            }
        }

        if (!matched) {
            new_pos++;
        }
    }

    return plan;
}

server_cache_reuse_routing_choice server_cache_reuse_select_slot(
        const std::vector<server_cache_reuse_routing_candidate> & candidates,
        const llama_tokens & tokens_new,
        size_t min_match) {
    server_cache_reuse_routing_choice choice;
    if (candidates.empty()) {
        return choice;
    }

    std::vector<server_cache_reuse_routing_choice> scores(candidates.size());
    bool has_overlap = false;

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & candidate = candidates[i];
        const llama_tokens * old_tokens = candidate.tokens_old;
        if (!old_tokens || min_match == 0) {
            continue;
        }

        const auto plan = server_cache_reuse_build_plan(
                *old_tokens,
                tokens_new,
                0,
                min_match,
                tokens_new.size());

        size_t sum_len = 0;
        size_t max_len = 0;
        for (const auto & block : plan) {
            sum_len += block.len;
            max_len = std::max(max_len, block.len);
        }

        scores[i].sum_len = sum_len;
        scores[i].max_len = max_len;
        has_overlap = has_overlap || (sum_len > 0);
    }

    if (has_overlap) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (scores[i].sum_len == 0) {
                continue;
            }

            if (choice.candidate_index == std::numeric_limits<size_t>::max()) {
                choice = scores[i];
                choice.candidate_index = i;
                continue;
            }

            const auto & candidate = candidates[i];
            const auto & best = candidates[choice.candidate_index];
            const auto & score = scores[i];
            const auto & best_score = scores[choice.candidate_index];

            const bool better =
                    (score.sum_len > best_score.sum_len) ||
                    (score.sum_len == best_score.sum_len && score.max_len > best_score.max_len) ||
                    (score.sum_len == best_score.sum_len && score.max_len == best_score.max_len &&
                        candidate.t_last_used < best.t_last_used) ||
                    (score.sum_len == best_score.sum_len && score.max_len == best_score.max_len &&
                        candidate.t_last_used == best.t_last_used && candidate.slot_id < best.slot_id);

            if (better) {
                choice = score;
                choice.candidate_index = i;
            }
        }
    } else {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (choice.candidate_index == std::numeric_limits<size_t>::max()) {
                choice.candidate_index = i;
                continue;
            }

            const auto & candidate = candidates[i];
            const auto & best = candidates[choice.candidate_index];

            if (candidate.t_last_used < best.t_last_used ||
                (candidate.t_last_used == best.t_last_used && candidate.slot_id < best.slot_id)) {
                choice.candidate_index = i;
            }
        }
    }

    return choice;
}
