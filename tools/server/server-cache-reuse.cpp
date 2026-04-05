#include "server-cache-reuse.h"

#include "common.h"

#include <algorithm>
#include <unordered_map>

static uint64_t server_cache_reuse_token_value(const llama_token token) {
    return (uint64_t) (uint32_t) token + 1ull;
}

static bool server_cache_reuse_match_seq(
        const llama_tokens & tokens,
        const size_t pos,
        const llama_tokens & seq) {
    if (seq.empty() || pos + seq.size() > tokens.size()) {
        return false;
    }

    for (size_t i = 0; i < seq.size(); ++i) {
        if (tokens[pos + i] != seq[i]) {
            return false;
        }
    }

    return true;
}

static bool server_cache_reuse_find_reasoning_end(
        const llama_tokens & tokens,
        const size_t start_pos,
        const llama_tokens & reasoning_start,
        const llama_tokens & reasoning_end,
        size_t & end_pos) {
    if (!server_cache_reuse_match_seq(tokens, start_pos, reasoning_start)) {
        return false;
    }

    size_t pos = start_pos + reasoning_start.size();
    while (pos < tokens.size()) {
        if (server_cache_reuse_match_seq(tokens, pos, reasoning_end)) {
            end_pos = pos + reasoning_end.size();
            return true;
        }
        ++pos;
    }

    return false;
}

struct server_cache_reuse_piece_cursor {
    size_t tok = 0;
    size_t off = 0;
};

struct server_cache_reuse_match_state {
    size_t old_tok = 0;
    size_t old_off = 0;
    size_t new_tok = 0;
    size_t new_off = 0;

    bool operator==(const server_cache_reuse_match_state & other) const {
        return old_tok == other.old_tok &&
               old_off == other.old_off &&
               new_tok == other.new_tok &&
               new_off == other.new_off;
    }
};

struct server_cache_reuse_match_state_hash {
    size_t operator()(const server_cache_reuse_match_state & s) const {
        size_t h = s.old_tok;
        h = h * 1315423911u + s.old_off;
        h = h * 1315423911u + s.new_tok;
        h = h * 1315423911u + s.new_off;
        return h;
    }
};

struct server_cache_reuse_match_result {
    bool ok = false;
    size_t best_new_tok = 0;
    size_t best_old_tok = 0;
    enum choice_type {
        STOP,
        ADVANCE,
        SKIP,
    } choice = STOP;
    size_t skip_resume_tok = std::numeric_limits<size_t>::max();
    size_t skip_resume_off = 0;
};

static void server_cache_reuse_normalize_cursor(
        const std::vector<std::string> & pieces,
        server_cache_reuse_piece_cursor & cur) {
    while (cur.tok < pieces.size()) {
        const auto & piece = pieces[cur.tok];
        if (cur.off < piece.size()) {
            break;
        }
        cur.tok++;
        cur.off = 0;
    }
}

static void server_cache_reuse_advance_cursor(
        const std::vector<std::string> & pieces,
        server_cache_reuse_piece_cursor & cur) {
    server_cache_reuse_normalize_cursor(pieces, cur);
    if (cur.tok >= pieces.size()) {
        return;
    }

    cur.off++;
    server_cache_reuse_normalize_cursor(pieces, cur);
}

static bool server_cache_reuse_is_whitespace_only(const std::string & piece) {
    if (piece.empty()) {
        return false;
    }

    for (const unsigned char ch : piece) {
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
            return false;
        }
    }

    return true;
}

static std::vector<std::string> server_cache_reuse_token_pieces(
        const struct llama_context * ctx,
        const llama_tokens & tokens) {
    std::vector<std::string> pieces;
    pieces.reserve(tokens.size());
    for (const auto token : tokens) {
        pieces.push_back(common_token_to_piece(ctx, token, true));
    }
    return pieces;
}

static std::vector<std::pair<size_t, size_t>> server_cache_reuse_merge_ranges(
        std::vector<std::pair<size_t, size_t>> ranges) {
    if (ranges.empty()) {
        return ranges;
    }

    std::sort(ranges.begin(), ranges.end());

    std::vector<std::pair<size_t, size_t>> merged;
    merged.reserve(ranges.size());
    merged.push_back(ranges[0]);

    for (size_t i = 1; i < ranges.size(); ++i) {
        auto & back = merged.back();
        if (ranges[i].first <= back.second) {
            back.second = std::max(back.second, ranges[i].second);
        } else {
            merged.push_back(ranges[i]);
        }
    }

    return merged;
}

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
        server_cache_reuse_reasoning_compaction & plan) {
    plan = {};
    (void) pieces_old_cache;

    if (tokens_old_raw.empty() ||
        tokens_old_cache.empty() ||
        tokens_new_raw.empty() ||
        reasoning_start.empty() ||
        reasoning_end.empty()) {
        return false;
    }

    std::vector<size_t> raw_to_cache = old_raw_to_cache_prefix;
    if (raw_to_cache.empty()) {
        raw_to_cache.resize(tokens_old_raw.size() + 1);
        for (size_t i = 0; i < raw_to_cache.size(); ++i) {
            raw_to_cache[i] = i;
        }
    }

    if (raw_to_cache.size() != tokens_old_raw.size() + 1) {
        return false;
    }

    std::vector<size_t> reasoning_end_at_start(tokens_old_raw.size(), std::numeric_limits<size_t>::max());
    std::vector<std::vector<server_cache_reuse_piece_cursor>> reasoning_resume_options(tokens_old_raw.size());
    for (size_t pos = 0; pos < tokens_old_raw.size(); ++pos) {
        size_t end_pos = 0;
        if (!server_cache_reuse_find_reasoning_end(tokens_old_raw, pos, reasoning_start, reasoning_end, end_pos)) {
            continue;
        }

        reasoning_end_at_start[pos] = end_pos;
        reasoning_resume_options[pos].push_back({ end_pos, 0 });

        size_t resume = end_pos;
        while (resume < pieces_old_raw.size() && server_cache_reuse_is_whitespace_only(pieces_old_raw[resume])) {
            for (size_t off = 1; off <= pieces_old_raw[resume].size(); ++off) {
                reasoning_resume_options[pos].push_back({ resume, off });
            }
            resume++;
            reasoning_resume_options[pos].push_back({ resume, 0 });
        }
    }

    std::unordered_map<server_cache_reuse_match_state, server_cache_reuse_match_result, server_cache_reuse_match_state_hash> memo;

    const auto can_skip_reasoning = [&](const server_cache_reuse_piece_cursor & old_cur,
                                        const server_cache_reuse_piece_cursor & new_cur) {
        if (old_cur.off != 0 || old_cur.tok >= tokens_old_raw.size()) {
            return false;
        }
        if (reasoning_end_at_start[old_cur.tok] == std::numeric_limits<size_t>::max()) {
            return false;
        }
        if (new_cur.off == 0 && server_cache_reuse_match_seq(tokens_new_raw, new_cur.tok, reasoning_start)) {
            return false;
        }
        return true;
    };

    const auto is_better = [](const server_cache_reuse_match_result & cand,
                              const server_cache_reuse_match_result & best) {
        if (!cand.ok) {
            return false;
        }
        if (!best.ok) {
            return true;
        }
        if (cand.best_new_tok != best.best_new_tok) {
            return cand.best_new_tok > best.best_new_tok;
        }
        return cand.best_old_tok > best.best_old_tok;
    };

    const auto dfs = [&](const auto & self,
                         server_cache_reuse_piece_cursor old_cur,
                         server_cache_reuse_piece_cursor new_cur) -> server_cache_reuse_match_result {
        struct chain_entry {
            server_cache_reuse_match_state state;
            bool at_boundary = false;
        };

        std::vector<chain_entry> chain;

        auto finish_chain = [&](server_cache_reuse_match_result result) {
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                server_cache_reuse_match_result cur;
                if (it->at_boundary) {
                    cur.ok = true;
                    cur.best_new_tok = it->state.new_tok;
                    cur.best_old_tok = it->state.old_tok;
                    cur.choice = server_cache_reuse_match_result::STOP;
                }

                if (is_better(result, cur)) {
                    cur = result;
                    cur.choice = server_cache_reuse_match_result::ADVANCE;
                    cur.skip_resume_tok = std::numeric_limits<size_t>::max();
                    cur.skip_resume_off = 0;
                }

                memo[it->state] = cur;
                result = cur;
            }

            return result;
        };

        while (true) {
            server_cache_reuse_normalize_cursor(pieces_old_raw, old_cur);
            server_cache_reuse_normalize_cursor(pieces_new_raw, new_cur);

            const server_cache_reuse_match_state state {
                old_cur.tok, old_cur.off, new_cur.tok, new_cur.off
            };

            if (auto it = memo.find(state); it != memo.end()) {
                return finish_chain(it->second);
            }

            const bool at_boundary = old_cur.off == 0 && new_cur.off == 0;
            server_cache_reuse_match_result result;
            if (at_boundary) {
                result.ok = true;
                result.best_new_tok = new_cur.tok;
                result.best_old_tok = old_cur.tok;
                result.choice = server_cache_reuse_match_result::STOP;
            }

            if (old_cur.tok >= tokens_old_raw.size()) {
                memo[state] = result;
                return finish_chain(result);
            }

            if (new_cur.tok >= tokens_new_raw.size()) {
                if (can_skip_reasoning(old_cur, new_cur)) {
                    for (const auto & resume_cur : reasoning_resume_options[old_cur.tok]) {
                        auto cand = self(self, resume_cur, new_cur);
                        if (is_better(cand, result)) {
                            result = cand;
                            result.choice = server_cache_reuse_match_result::SKIP;
                            result.skip_resume_tok = resume_cur.tok;
                            result.skip_resume_off = resume_cur.off;
                        }
                    }
                }

                memo[state] = result;
                return finish_chain(result);
            }

            const auto & old_piece = pieces_old_raw[old_cur.tok];
            const auto & new_piece = pieces_new_raw[new_cur.tok];

            const bool can_advance =
                    !can_skip_reasoning(old_cur, new_cur) &&
                    old_cur.off < old_piece.size() &&
                    new_cur.off < new_piece.size() &&
                    old_piece[old_cur.off] == new_piece[new_cur.off];

            if (!can_advance) {
                if (can_skip_reasoning(old_cur, new_cur)) {
                    for (const auto & resume_cur : reasoning_resume_options[old_cur.tok]) {
                        auto cand = self(self, resume_cur, new_cur);
                        if (is_better(cand, result)) {
                            result = cand;
                            result.choice = server_cache_reuse_match_result::SKIP;
                            result.skip_resume_tok = resume_cur.tok;
                            result.skip_resume_off = resume_cur.off;
                        }
                    }
                }

                memo[state] = result;
                return finish_chain(result);
            }

            chain.push_back({ state, at_boundary });
            server_cache_reuse_advance_cursor(pieces_old_raw, old_cur);
            server_cache_reuse_advance_cursor(pieces_new_raw, new_cur);
        }
    };

    std::vector<std::pair<size_t, size_t>> raw_delete_ranges;
    server_cache_reuse_piece_cursor old_cur;
    server_cache_reuse_piece_cursor new_cur;
    auto root = dfs(dfs, old_cur, new_cur);
    if (!root.ok) {
        return false;
    }

    size_t old_stop_tok = 0;
    size_t new_stop_tok = 0;
    std::vector<size_t> raw_to_old_raw_prefix(tokens_new_raw.size() + 1, 0);
    raw_to_old_raw_prefix[0] = 0;

    while (true) {
        server_cache_reuse_normalize_cursor(pieces_old_raw, old_cur);
        server_cache_reuse_normalize_cursor(pieces_new_raw, new_cur);

        const server_cache_reuse_match_state state {
            old_cur.tok, old_cur.off, new_cur.tok, new_cur.off
        };
        const auto it = memo.find(state);
        if (it == memo.end() || !it->second.ok) {
            return false;
        }

        if (it->second.choice == server_cache_reuse_match_result::STOP) {
            old_stop_tok = old_cur.tok;
            new_stop_tok = new_cur.tok;
            break;
        }

        if (it->second.choice == server_cache_reuse_match_result::SKIP) {
            raw_delete_ranges.emplace_back(old_cur.tok, it->second.skip_resume_tok);
            old_cur = { it->second.skip_resume_tok, it->second.skip_resume_off };
            continue;
        }

        if (it->second.choice != server_cache_reuse_match_result::ADVANCE) {
            return false;
        }

        server_cache_reuse_advance_cursor(pieces_old_raw, old_cur);
        server_cache_reuse_advance_cursor(pieces_new_raw, new_cur);

        if (new_cur.off == 0 && new_cur.tok < raw_to_old_raw_prefix.size()) {
            raw_to_old_raw_prefix[new_cur.tok] = old_cur.tok + (old_cur.off != 0 ? 1 : 0);
        }
    }

    if (raw_delete_ranges.empty()) {
        return false;
    }

    plan.raw_prefix_len = new_stop_tok;
    if (plan.raw_prefix_len == 0) {
        return false;
    }

    std::vector<std::pair<size_t, size_t>> cache_delete_ranges;
    cache_delete_ranges.reserve(raw_delete_ranges.size());
    for (const auto & range : raw_delete_ranges) {
        const size_t cache_lo = raw_to_cache[range.first];
        const size_t cache_hi = raw_to_cache[range.second];
        if (cache_hi < cache_lo || cache_hi > tokens_old_cache.size()) {
            return false;
        }
        if (cache_lo < cache_hi) {
            cache_delete_ranges.emplace_back(cache_lo, cache_hi);
        }
    }

    cache_delete_ranges = server_cache_reuse_merge_ranges(std::move(cache_delete_ranges));

    const size_t cache_stop_tok = raw_to_cache[old_stop_tok];
    if (cache_stop_tok > tokens_old_cache.size()) {
        return false;
    }

    plan.cache_prefix_tokens.reserve(cache_stop_tok);
    size_t cache_pos = 0;
    for (const auto & range : cache_delete_ranges) {
        if (range.first >= cache_stop_tok) {
            break;
        }
        for (; cache_pos < range.first; ++cache_pos) {
            plan.cache_prefix_tokens.push_back(tokens_old_cache[cache_pos]);
        }
        cache_pos = std::min(range.second, cache_stop_tok);
    }
    for (; cache_pos < cache_stop_tok; ++cache_pos) {
        plan.cache_prefix_tokens.push_back(tokens_old_cache[cache_pos]);
    }

    if (plan.cache_prefix_tokens.empty()) {
        return false;
    }

    plan.raw_to_cache_prefix.assign(plan.raw_prefix_len + 1, 0);
    plan.raw_to_cache_prefix[0] = 0;

    size_t deleted_before = 0;
    size_t delete_idx = 0;

    for (size_t raw_tok = 1; raw_tok <= plan.raw_prefix_len; ++raw_tok) {
        size_t old_raw_prefix = raw_to_old_raw_prefix[raw_tok];
        old_raw_prefix = std::min(old_raw_prefix, tokens_old_raw.size());

        const size_t old_cache_prefix = raw_to_cache[old_raw_prefix];

        while (delete_idx < cache_delete_ranges.size() &&
               cache_delete_ranges[delete_idx].second <= old_cache_prefix) {
            deleted_before += cache_delete_ranges[delete_idx].second - cache_delete_ranges[delete_idx].first;
            delete_idx++;
        }

        size_t deleted_partial = 0;
        if (delete_idx < cache_delete_ranges.size() &&
            cache_delete_ranges[delete_idx].first < old_cache_prefix) {
            deleted_partial = old_cache_prefix - cache_delete_ranges[delete_idx].first;
        }

        if (old_cache_prefix < deleted_before + deleted_partial) {
            return false;
        }

        const size_t compact_prefix = old_cache_prefix - deleted_before - deleted_partial;
        if (compact_prefix > plan.cache_prefix_tokens.size()) {
            return false;
        }

        plan.raw_to_cache_prefix[raw_tok] = compact_prefix;
    }

    if (plan.raw_to_cache_prefix.back() != plan.cache_prefix_tokens.size()) {
        return false;
    }

    return true;
}

bool server_cache_reuse_build_reasoning_compaction(
        const struct llama_context * ctx,
        const llama_tokens & tokens_old_raw,
        const llama_tokens & tokens_old_cache,
        const std::vector<size_t> & old_raw_to_cache_prefix,
        const llama_tokens & tokens_new_raw,
        const llama_tokens & reasoning_start,
        const llama_tokens & reasoning_end,
        server_cache_reuse_reasoning_compaction & plan) {
    if (ctx == nullptr) {
        return false;
    }

    const auto pieces_old_raw = server_cache_reuse_token_pieces(ctx, tokens_old_raw);
    const auto pieces_old_cache = server_cache_reuse_token_pieces(ctx, tokens_old_cache);
    const auto pieces_new_raw = server_cache_reuse_token_pieces(ctx, tokens_new_raw);

    return server_cache_reuse_build_reasoning_compaction_pieces(
            tokens_old_raw,
            pieces_old_raw,
            tokens_old_cache,
            pieces_old_cache,
            old_raw_to_cache_prefix,
            tokens_new_raw,
            pieces_new_raw,
            reasoning_start,
            reasoning_end,
            plan);
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

                server_cache_reuse_block block;
                block.old_pos = old_pos;
                block.new_pos = new_pos;
                block.len     = len;
                plan.push_back(std::move(block));
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

    if (!plan.empty()) {
        auto & last = plan.back();
        if (last.new_pos + last.len == new_limit) {
            if (last.len > 0) {
                last.len--;
            }
            if (last.len == 0) {
                plan.pop_back();
            }
        }
    }

    return plan;
}

std::vector<server_cache_reuse_block> server_cache_reuse_build_mapped_plan(
        const llama_tokens & tokens_old_raw,
        const llama_tokens & tokens_old_cache,
        const std::vector<size_t> & old_raw_to_cache_prefix,
        const llama_tokens & tokens_new_raw,
        size_t start_pos,
        size_t min_match,
        size_t new_limit) {
    auto plan = server_cache_reuse_build_plan(
            tokens_old_raw,
            tokens_new_raw,
            start_pos,
            min_match,
            new_limit);

    std::vector<size_t> raw_to_cache = old_raw_to_cache_prefix;
    if (raw_to_cache.empty()) {
        raw_to_cache.resize(tokens_old_raw.size() + 1);
        for (size_t i = 0; i < raw_to_cache.size(); ++i) {
            raw_to_cache[i] = i;
        }
    }

    if (raw_to_cache.size() != tokens_old_raw.size() + 1) {
        return {};
    }

    for (auto & block : plan) {
        const size_t old_end = block.old_pos + block.len;
        if (old_end > tokens_old_raw.size()) {
            return {};
        }

        const size_t cache_lo = raw_to_cache[block.old_pos];
        const size_t cache_hi = raw_to_cache[old_end];
        if (cache_hi < cache_lo || cache_hi > tokens_old_cache.size()) {
            return {};
        }

        block.old_cache_pos = cache_lo;
        block.len_cache = cache_hi - cache_lo;
        block.cache_tokens.assign(tokens_old_cache.begin() + cache_lo, tokens_old_cache.begin() + cache_hi);
        block.raw_to_cache_delta.resize(block.len + 1);
        for (size_t i = 0; i <= block.len; ++i) {
            const size_t cur_cache = raw_to_cache[block.old_pos + i];
            if (cur_cache < cache_lo) {
                return {};
            }
            block.raw_to_cache_delta[i] = cur_cache - cache_lo;
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
