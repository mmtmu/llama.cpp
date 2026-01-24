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

    if (t.failures > 0) {
        return 1;
    }
    return 0;
}
