#include "testing.h"

#include "server-cache-reuse.h"

static void assert_block(testing & t, const server_cache_reuse_block & block, size_t old_pos, size_t new_pos, size_t len) {
    t.assert_true(block.old_pos == old_pos);
    t.assert_true(block.new_pos == new_pos);
    t.assert_true(block.len     == len);
}

static void assert_prefix_map(testing & t, const std::vector<size_t> & got, const std::vector<size_t> & expected) {
    t.assert_true(got.size() == expected.size());
    if (got.size() != expected.size()) {
        return;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        t.assert_true(got[i] == expected[i]);
    }
}

int main() {
    testing t;

    t.test("identical prompt", [](testing & t) {
        llama_tokens old_tokens = { 1, 2, 3, 4, 5 };
        llama_tokens new_tokens = { 1, 2, 3, 4, 5 };

        auto plan = server_cache_reuse_build_plan(old_tokens, new_tokens, 0, 2, new_tokens.size());
        t.assert_true(plan.size() == 1);
        assert_block(t, plan[0], 0, 0, 5);
    });

    t.test("deletion in prefix", [](testing & t) {
        llama_tokens old_tokens = { 1, 2, 3, 4, 5, 6, 7 };
        llama_tokens new_tokens = { 1, 2, 4, 5, 6, 7 };

        auto plan = server_cache_reuse_build_plan(old_tokens, new_tokens, 2, 2, new_tokens.size());
        t.assert_true(plan.size() == 1);
        assert_block(t, plan[0], 3, 2, 4);
    });

    t.test("replacement in prefix", [](testing & t) {
        llama_tokens old_tokens = { 1, 2, 3, 4, 5, 6 };
        llama_tokens new_tokens = { 1, 2, 9, 4, 5, 6 };

        auto plan = server_cache_reuse_build_plan(old_tokens, new_tokens, 2, 2, new_tokens.size());
        t.assert_true(plan.size() == 1);
        assert_block(t, plan[0], 3, 3, 3);
    });

    t.test("prefix replaced, suffix added", [](testing & t) {
        llama_tokens old_tokens = { 1, 2, 3, 4, 5 };
        llama_tokens new_tokens = { 9, 3, 4, 5, 6 };

        auto plan = server_cache_reuse_build_plan(old_tokens, new_tokens, 0, 2, new_tokens.size());
        t.assert_true(plan.size() == 1);
        assert_block(t, plan[0], 2, 1, 3);
    });

    t.test("repeated chunk chooses earliest", [](testing & t) {
        llama_tokens old_tokens = { 7, 7, 1, 2, 3, 1, 2, 3, 4 };
        llama_tokens new_tokens = { 1, 2, 3 };

        auto plan = server_cache_reuse_build_plan(old_tokens, new_tokens, 0, 3, new_tokens.size());
        t.assert_true(plan.size() == 1);
        assert_block(t, plan[0], 2, 0, 3);
    });

    t.test("below threshold", [](testing & t) {
        llama_tokens old_tokens = { 1, 2, 3, 4 };
        llama_tokens new_tokens = { 1, 2, 3, 9 };

        auto plan = server_cache_reuse_build_plan(old_tokens, new_tokens, 0, 4, new_tokens.size());
        t.assert_true(plan.empty());
    });

    t.test("routing prefers larger island sum over better LCP", [](testing & t) {
        llama_tokens new_tokens = { 9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 10 };
        llama_tokens old_a      = { 9, 9, 1, 20, 21, 22, 23, 24, 25, 26, 27, 10 }; // better LCP
        llama_tokens old_b      = { 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11 };             // larger reusable island

        std::vector<server_cache_reuse_routing_candidate> candidates = {
            { &old_a, 1, 0 },
            { &old_b, 2, 1 },
        };

        const auto choice = server_cache_reuse_select_slot(candidates, new_tokens, 4);
        t.assert_true(choice.candidate_index == 1);
        t.assert_true(choice.sum_len == 8);
        t.assert_true(choice.max_len == 8);
    });

    t.test("routing falls back to LRU when all overlaps are zero", [](testing & t) {
        llama_tokens new_tokens = { 1, 2, 3, 4 };
        llama_tokens old_a      = { 7, 8, 9, 10 };
        llama_tokens old_b      = { 11, 12, 13, 14 };

        std::vector<server_cache_reuse_routing_candidate> candidates = {
            { &old_a, 50, 0 },
            { &old_b, 10, 1 },
        };

        const auto choice = server_cache_reuse_select_slot(candidates, new_tokens, 2);
        t.assert_true(choice.candidate_index == 1);
        t.assert_true(choice.sum_len == 0);
        t.assert_true(choice.max_len == 0);
    });

    t.test("routing tie-breaks are deterministic", [](testing & t) {
        llama_tokens new_tokens = { 1, 2, 3, 4, 5, 6 };
        llama_tokens old_small  = { 1, 2, 9, 9, 5, 6 }; // sum=4, max=2
        llama_tokens old_big_a  = { 0, 1, 2, 3, 4, 0 }; // sum=4, max=4
        llama_tokens old_big_b  = { 8, 1, 2, 3, 4, 8 }; // sum=4, max=4
        llama_tokens old_big_c  = { 7, 1, 2, 3, 4, 7 }; // sum=4, max=4

        // max_len tie-break beats LRU
        {
            std::vector<server_cache_reuse_routing_candidate> candidates = {
                { &old_small, 1, 0 }, // older, but smaller max_len
                { &old_big_a, 2, 1 },
            };

            const auto choice = server_cache_reuse_select_slot(candidates, new_tokens, 2);
            t.assert_true(choice.candidate_index == 1);
        }

        // with equal scores, older LRU wins
        {
            std::vector<server_cache_reuse_routing_candidate> candidates = {
                { &old_big_a, 20, 4 },
                { &old_big_b, 10, 5 },
            };

            const auto choice = server_cache_reuse_select_slot(candidates, new_tokens, 2);
            t.assert_true(choice.candidate_index == 1);
        }

        // with equal score and equal LRU timestamp, lower slot_id wins
        {
            std::vector<server_cache_reuse_routing_candidate> candidates = {
                { &old_big_a, 10, 7 },
                { &old_big_c, 10, 3 },
            };

            const auto choice = server_cache_reuse_select_slot(candidates, new_tokens, 2);
            t.assert_true(choice.candidate_index == 1);
        }
    });

    t.test("reasoning compaction preserves cache seam tokens", [](testing & t) {
        const llama_tokens old_raw = { 1, 10, 50, 2, 51, 10, 60 };
        const std::vector<std::string> pieces_old_raw = {
            "A", "\n", "<think>", "reason", "</think>", "\n", "<tool>",
        };

        const llama_tokens old_cache = old_raw;
        const std::vector<std::string> pieces_old_cache = pieces_old_raw;

        const llama_tokens new_raw = { 1, 367, 60, 99 };
        const std::vector<std::string> pieces_new_raw = {
            "A", "\n\n", "<tool>", " tail",
        };

        const std::vector<size_t> old_map = { 0, 1, 2, 3, 4, 5, 6, 7 };
        server_cache_reuse_reasoning_compaction plan;

        const bool ok = server_cache_reuse_build_reasoning_compaction_pieces(
                old_raw,
                pieces_old_raw,
                old_cache,
                pieces_old_cache,
                old_map,
                new_raw,
                pieces_new_raw,
                { 50 },
                { 51 },
                plan);

        t.assert_true(ok);
        if (!ok) {
            return;
        }
        t.assert_true(plan.raw_prefix_len == 3);
        t.assert_true(plan.cache_prefix_tokens == llama_tokens({ 1, 10, 10, 60 }));
        assert_prefix_map(t, plan.raw_to_cache_prefix, { 0, 1, 3, 4 });
    });

    t.test("reasoning compaction reuses existing divergent cache seam", [](testing & t) {
        const llama_tokens old_raw = { 1, 367, 50, 2, 51, 60 };
        const std::vector<std::string> pieces_old_raw = {
            "A", "\n\n", "<think>", "reason", "</think>", "<tool>",
        };

        const llama_tokens old_cache = { 1, 10, 10, 50, 2, 51, 60 };
        const std::vector<std::string> pieces_old_cache = {
            "A", "\n", "\n", "<think>", "reason", "</think>", "<tool>",
        };

        const llama_tokens new_raw = { 1, 367, 60, 99 };
        const std::vector<std::string> pieces_new_raw = {
            "A", "\n\n", "<tool>", " tail",
        };

        const std::vector<size_t> old_map = { 0, 1, 3, 4, 5, 6, 7 };
        server_cache_reuse_reasoning_compaction plan;

        const bool ok = server_cache_reuse_build_reasoning_compaction_pieces(
                old_raw,
                pieces_old_raw,
                old_cache,
                pieces_old_cache,
                old_map,
                new_raw,
                pieces_new_raw,
                { 50 },
                { 51 },
                plan);

        t.assert_true(ok);
        if (!ok) {
            return;
        }
        t.assert_true(plan.raw_prefix_len == 3);
        t.assert_true(plan.cache_prefix_tokens == llama_tokens({ 1, 10, 10, 60 }));
        assert_prefix_map(t, plan.raw_to_cache_prefix, { 0, 1, 3, 4 });
    });

    t.test("reasoning compaction keeps oversized cache whitespace seam", [](testing & t) {
        const llama_tokens old_raw = { 1, 10, 50, 10, 2, 10, 51, 4368, 60 };
        const std::vector<std::string> pieces_old_raw = {
            "A", "\n", "<think>", "\n", "reason", "\n", "</think>", "\n\n\n", "<tool>",
        };

        const llama_tokens new_raw = { 1, 367, 60, 99 };
        const std::vector<std::string> pieces_new_raw = {
            "A", "\n\n", "<tool>", " tail",
        };

        const std::vector<size_t> old_map = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        server_cache_reuse_reasoning_compaction plan;

        const bool ok = server_cache_reuse_build_reasoning_compaction_pieces(
                old_raw,
                pieces_old_raw,
                old_raw,
                pieces_old_raw,
                old_map,
                new_raw,
                pieces_new_raw,
                { 50 },
                { 51 },
                plan);

        t.assert_true(ok);
        if (!ok) {
            return;
        }
        t.assert_true(plan.raw_prefix_len == 3);
        t.assert_true(plan.cache_prefix_tokens == llama_tokens({ 1, 10, 4368, 60 }));
        assert_prefix_map(t, plan.raw_to_cache_prefix, { 0, 1, 3, 4 });
    });

    t.test("reasoning compaction rejects non-reasoning mismatch", [](testing & t) {
        const llama_tokens old_raw = { 1, 10, 50, 2, 51, 10, 60 };
        const std::vector<std::string> pieces_old_raw = {
            "A", "\n", "<think>", "reason", "</think>", "\n", "<tool>",
        };

        server_cache_reuse_reasoning_compaction plan;
        const bool ok = server_cache_reuse_build_reasoning_compaction_pieces(
                old_raw,
                pieces_old_raw,
                old_raw,
                pieces_old_raw,
                { 0, 1, 2, 3, 4, 5, 6, 7 },
                { 1, 88, 60 },
                { "A", "X", "<tool>" },
                { 50 },
                { 51 },
                plan);

        t.assert_true(!ok);
    });

    t.test("reasoning compaction handles many interleaved reasoning blocks", [](testing & t) {
        llama_tokens old_raw = { 1 };
        llama_tokens new_raw = { 1 };
        std::vector<std::string> pieces_old_raw = { "START" };
        std::vector<std::string> pieces_new_raw = { "START" };
        std::vector<size_t> old_map = { 0 };

        constexpr int n_sections = 15;
        for (int i = 0; i < n_sections; ++i) {
            const llama_token header_tok = 1000 + i;
            const llama_token reason_tok = 2000 + i;
            const llama_token tool_tok   = 3000 + i;
            const llama_token tool_body  = 4000 + i;

            // old raw/cache keeps the historical reasoning block
            old_raw.insert(old_raw.end(), { header_tok, 10, 50, 10, reason_tok, 10, 51, 4368, tool_tok, tool_body });
            pieces_old_raw.insert(pieces_old_raw.end(), {
                std::string("<ai:") + std::to_string(i) + ">",
                "\n",
                "<think>",
                "\n",
                std::string("reason-") + std::to_string(i),
                "\n",
                "</think>",
                "\n\n\n",
                "<minimax:tool_call>",
                std::string("<invoke:") + std::to_string(i) + ">",
            });

            // new raw drops the reasoning block and retokenizes the seam as a double newline
            new_raw.insert(new_raw.end(), { header_tok, 367, tool_tok, tool_body });
            pieces_new_raw.insert(pieces_new_raw.end(), {
                std::string("<ai:") + std::to_string(i) + ">",
                "\n\n",
                "<minimax:tool_call>",
                std::string("<invoke:") + std::to_string(i) + ">",
            });
        }

        // old cached prompt can include a trailing generated assistant tail from the
        // previous request that is not part of the new request history.
        old_raw.insert(old_raw.end(), { 7000, 10, 50, 10, 7100 });
        pieces_old_raw.insert(pieces_old_raw.end(), {
            "<gen-assistant>",
            "\n",
            "<think>",
            "\n",
            "tail-token",
        });

        new_raw.push_back(9000);
        pieces_new_raw.push_back("<new-user>");

        old_map.resize(old_raw.size() + 1);
        for (size_t i = 0; i < old_map.size(); ++i) {
            old_map[i] = i;
        }

        server_cache_reuse_reasoning_compaction plan;
        const bool ok = server_cache_reuse_build_reasoning_compaction_pieces(
                old_raw,
                pieces_old_raw,
                old_raw,
                pieces_old_raw,
                old_map,
                new_raw,
                pieces_new_raw,
                { 50 },
                { 51 },
                plan);

        t.assert_true(ok);
        if (!ok) {
            return;
        }
        t.assert_true(plan.raw_prefix_len == new_raw.size() - 1);
        t.assert_true(plan.cache_prefix_tokens.size() == 1 + n_sections * 5);
        t.assert_true(plan.raw_to_cache_prefix.size() == plan.raw_prefix_len + 1);
        t.assert_true(plan.raw_to_cache_prefix.front() == 0);
        t.assert_true(plan.raw_to_cache_prefix.back() == plan.cache_prefix_tokens.size());

        size_t raw_pos = 1;
        size_t cache_pos = 1;
        for (int i = 0; i < n_sections; ++i) {
            t.assert_true(plan.raw_to_cache_prefix[raw_pos + 0] == cache_pos + 0);
            t.assert_true(plan.raw_to_cache_prefix[raw_pos + 1] == cache_pos + 1);
            t.assert_true(plan.raw_to_cache_prefix[raw_pos + 2] == cache_pos + 3);
            t.assert_true(plan.raw_to_cache_prefix[raw_pos + 3] == cache_pos + 4);
            t.assert_true(plan.raw_to_cache_prefix[raw_pos + 4] == cache_pos + 5);
            raw_pos += 4;
            cache_pos += 5;
        }
    });

    t.test("reasoning compaction handles long matching prefixes", [](testing & t) {
        llama_tokens old_raw = { 1 };
        llama_tokens new_raw = { 1 };
        std::vector<std::string> pieces_old_raw = { "START" };
        std::vector<std::string> pieces_new_raw = { "START" };

        constexpr int n_prefix = 20000;
        for (int i = 0; i < n_prefix; ++i) {
            old_raw.push_back(10000 + i);
            new_raw.push_back(10000 + i);
            pieces_old_raw.push_back(std::string("tok-") + std::to_string(i));
            pieces_new_raw.push_back(std::string("tok-") + std::to_string(i));
        }

        old_raw.insert(old_raw.end(), { 10, 50, 10, 200000, 10, 51, 4368, 60 });
        pieces_old_raw.insert(pieces_old_raw.end(), {
            "\n", "<think>", "\n", "reason", "\n", "</think>", "\n\n\n", "<tool>",
        });

        new_raw.insert(new_raw.end(), { 367, 60, 99 });
        pieces_new_raw.insert(pieces_new_raw.end(), {
            "\n\n", "<tool>", " tail",
        });

        std::vector<size_t> old_map(old_raw.size() + 1);
        for (size_t i = 0; i < old_map.size(); ++i) {
            old_map[i] = i;
        }

        server_cache_reuse_reasoning_compaction plan;
        const bool ok = server_cache_reuse_build_reasoning_compaction_pieces(
                old_raw,
                pieces_old_raw,
                old_raw,
                pieces_old_raw,
                old_map,
                new_raw,
                pieces_new_raw,
                { 50 },
                { 51 },
                plan);

        t.assert_true(ok);
        if (!ok) {
            return;
        }
        t.assert_true(plan.raw_prefix_len == new_raw.size() - 1);
        t.assert_true(plan.cache_prefix_tokens.size() == (size_t) n_prefix + 4);
        t.assert_true(plan.raw_to_cache_prefix.back() == plan.cache_prefix_tokens.size());
    });

    if (t.failures > 0) {
        return 1;
    }
    return 0;
}
