#include "testing.h"

#include "server-cache-reuse.h"

static void assert_block(testing & t, const server_cache_reuse_block & block, size_t old_pos, size_t new_pos, size_t len) {
    t.assert_true(block.old_pos == old_pos);
    t.assert_true(block.new_pos == new_pos);
    t.assert_true(block.len     == len);
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

    if (t.failures > 0) {
        return 1;
    }
    return 0;
}
