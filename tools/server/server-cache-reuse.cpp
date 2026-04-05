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
    size_t skip_resume_tok = std::numeric_limits<size_t>::max();
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
    std::vector<std::vector<size_t>> reasoning_resume_options(tokens_old_raw.size());
    for (size_t pos = 0; pos < tokens_old_raw.size(); ++pos) {
        size_t end_pos = 0;
        if (!server_cache_reuse_find_reasoning_end(tokens_old_raw, pos, reasoning_start, reasoning_end, end_pos)) {
            continue;
        }

        reasoning_end_at_start[pos] = end_pos;
        reasoning_resume_options[pos].push_back(end_pos);

        size_t resume = end_pos;
        while (resume < pieces_old_raw.size() && server_cache_reuse_is_whitespace_only(pieces_old_raw[resume])) {
            resume++;
            reasoning_resume_options[pos].push_back(resume);
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

    const auto dfs = [&](const auto & self,
                         server_cache_reuse_piece_cursor old_cur,
                         server_cache_reuse_piece_cursor new_cur) -> bool {
        server_cache_reuse_normalize_cursor(pieces_old_raw, old_cur);
        server_cache_reuse_normalize_cursor(pieces_new_raw, new_cur);

        const server_cache_reuse_match_state state {
            old_cur.tok, old_cur.off, new_cur.tok, new_cur.off
        };

        if (auto it = memo.find(state); it != memo.end()) {
            return it->second.ok;
        }

        auto & result = memo[state];

        if (old_cur.tok >= tokens_old_raw.size()) {
            result.ok = (new_cur.off == 0);
            return result.ok;
        }

        if (new_cur.tok >= tokens_new_raw.size()) {
            if (can_skip_reasoning(old_cur, new_cur)) {
                for (const size_t resume_tok : reasoning_resume_options[old_cur.tok]) {
                    if (self(self, { resume_tok, 0 }, new_cur)) {
                        result.ok = true;
                        result.skip_resume_tok = resume_tok;
                        return true;
                    }
                }
            }

            result.ok = false;
            return false;
        }

        const auto & old_piece = pieces_old_raw[old_cur.tok];
        const auto & new_piece = pieces_new_raw[new_cur.tok];

        if (old_cur.off < old_piece.size() &&
            new_cur.off < new_piece.size() &&
            old_piece[old_cur.off] == new_piece[new_cur.off]) {
            auto next_old = old_cur;
            auto next_new = new_cur;
            server_cache_reuse_advance_cursor(pieces_old_raw, next_old);
            server_cache_reuse_advance_cursor(pieces_new_raw, next_new);
            if (self(self, next_old, next_new)) {
                result.ok = true;
                return true;
            }
        }

        if (can_skip_reasoning(old_cur, new_cur)) {
            for (const size_t resume_tok : reasoning_resume_options[old_cur.tok]) {
                if (self(self, { resume_tok, 0 }, new_cur)) {
                    result.ok = true;
                    result.skip_resume_tok = resume_tok;
                    return true;
                }
            }
        }

        result.ok = false;
        return false;
    };

    std::vector<std::pair<size_t, size_t>> raw_delete_ranges;
    server_cache_reuse_piece_cursor old_cur;
    server_cache_reuse_piece_cursor new_cur;
    if (!dfs(dfs, old_cur, new_cur)) {
        return false;
    }

    while (true) {
        server_cache_reuse_normalize_cursor(pieces_old_raw, old_cur);
        server_cache_reuse_normalize_cursor(pieces_new_raw, new_cur);

        if (old_cur.tok >= tokens_old_raw.size()) {
            break;
        }

        const server_cache_reuse_match_state state {
            old_cur.tok, old_cur.off, new_cur.tok, new_cur.off
        };
        const auto it = memo.find(state);
        if (it == memo.end() || !it->second.ok) {
            return false;
        }

        if (it->second.skip_resume_tok != std::numeric_limits<size_t>::max()) {
            raw_delete_ranges.emplace_back(old_cur.tok, it->second.skip_resume_tok);
            old_cur = { it->second.skip_resume_tok, 0 };
            continue;
        }

        server_cache_reuse_advance_cursor(pieces_old_raw, old_cur);
        server_cache_reuse_advance_cursor(pieces_new_raw, new_cur);
    }

    if (raw_delete_ranges.empty() || new_cur.off != 0) {
        return false;
    }

    plan.raw_prefix_len = new_cur.tok;

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

    plan.cache_prefix_tokens.reserve(tokens_old_cache.size());
    size_t cache_pos = 0;
    for (const auto & range : cache_delete_ranges) {
        for (; cache_pos < range.first; ++cache_pos) {
            plan.cache_prefix_tokens.push_back(tokens_old_cache[cache_pos]);
        }
        cache_pos = range.second;
    }
    for (; cache_pos < tokens_old_cache.size(); ++cache_pos) {
        plan.cache_prefix_tokens.push_back(tokens_old_cache[cache_pos]);
    }

    if (plan.cache_prefix_tokens.empty()) {
        return false;
    }

    plan.raw_to_cache_prefix.assign(plan.raw_prefix_len + 1, 0);

    server_cache_reuse_piece_cursor raw_prefix_cur;
    server_cache_reuse_piece_cursor cache_cur;
    server_cache_reuse_normalize_cursor(pieces_new_raw, raw_prefix_cur);
    server_cache_reuse_normalize_cursor(pieces_old_cache, cache_cur);

    std::vector<std::string> pieces_cache_prefix;
    pieces_cache_prefix.reserve(plan.cache_prefix_tokens.size());
    cache_pos = 0;
    for (const auto & range : cache_delete_ranges) {
        for (; cache_pos < range.first; ++cache_pos) {
            pieces_cache_prefix.push_back(pieces_old_cache[cache_pos]);
        }
        cache_pos = range.second;
    }
    for (; cache_pos < pieces_old_cache.size(); ++cache_pos) {
        pieces_cache_prefix.push_back(pieces_old_cache[cache_pos]);
    }

    cache_cur = {};
    server_cache_reuse_normalize_cursor(pieces_cache_prefix, cache_cur);
    raw_prefix_cur = {};
    server_cache_reuse_normalize_cursor(pieces_new_raw, raw_prefix_cur);

    while (raw_prefix_cur.tok < plan.raw_prefix_len) {
        if (cache_cur.tok >= pieces_cache_prefix.size()) {
            return false;
        }

        const auto & raw_piece = pieces_new_raw[raw_prefix_cur.tok];
        const auto & cache_piece = pieces_cache_prefix[cache_cur.tok];

        if (raw_prefix_cur.off >= raw_piece.size() || cache_cur.off >= cache_piece.size()) {
            return false;
        }

        if (raw_piece[raw_prefix_cur.off] != cache_piece[cache_cur.off]) {
            return false;
        }

        server_cache_reuse_advance_cursor(pieces_new_raw, raw_prefix_cur);
        server_cache_reuse_advance_cursor(pieces_cache_prefix, cache_cur);

        if (raw_prefix_cur.off == 0 && raw_prefix_cur.tok <= plan.raw_prefix_len) {
            plan.raw_to_cache_prefix[raw_prefix_cur.tok] = cache_cur.tok;
        }
    }

    server_cache_reuse_normalize_cursor(pieces_cache_prefix, cache_cur);
    if (cache_cur.tok != pieces_cache_prefix.size() || cache_cur.off != 0) {
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
