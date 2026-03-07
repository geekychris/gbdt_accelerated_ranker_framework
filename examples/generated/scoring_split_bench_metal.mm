#include "scoring_split_metal_core.h"
#include <time.h>

static const float FEAT_MIN[15] = {
    0.00f, 0.00f, 177.40f, 0.00f, 0.20f, 0.00f, 0.00f, 0.20f,
    0.80f, 0.00f, 1.80f, -0.77f, -0.68f, 0.05f, 0.12f,
};
static const float FEAT_MAX[15] = {
    350562.76f, 1008308.61f, 223.60f, 2091.66f, 33.80f, 0.27f, 0.34f, 5.80f,
    2.20f, 1984.60f, 31.20f, 0.86f, 0.46f, 1.06f, 1.12f,
};

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float randf_range(unsigned int *state, float lo, float hi) {
    *state = *state * 1103515245u + 12345u;
    float t = (float)(*state >> 16) / 65535.0f;
    return lo + t * (hi - lo);
}

int main(int argc, char** argv) {
    @autoreleasepool {
    int n_items    = 1000000;
    int topk       = 100;
    int warmup     = 3;
    int iterations = 20;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_items    = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) topk       = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) warmup     = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) seed       = (unsigned)atoi(argv[++i]);
        else {
            fprintf(stderr,
                "Usage: %s [-n items] [-k topK] [-w warmup] [-i iters] [-s seed]\n"
                "\n  Defaults: -n 1000000 -k 100 -w 3 -i 20 -s 42\n", argv[0]);
            return (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) ? 0 : 1;
        }
    }

    printf("=== Metal (Apple Silicon GPU) Split-Feature Scoring Benchmark ===\n");
    printf("  Items:         %d\n", n_items);
    printf("  Top-K:         %d\n", topk);
    printf("  User features: %d\n", NUM_USER_FEATURES);
    printf("  Item features: %d\n", NUM_ITEM_FEATURES);
    printf("  Trees:         %d\n", NUM_TREES);
    printf("  Warmup:        %d\n", warmup);
    printf("  Iterations:    %d\n\n", iterations);

    /* Generate random data in realistic feature ranges */
    unsigned int rng = seed;
    float user[NUM_USER_FEATURES];
    for (int i = 0; i < NUM_USER_FEATURES; i++)
        user[i] = randf_range(&rng, FEAT_MIN[i], FEAT_MAX[i]);

    size_t items_bytes = (size_t)n_items * NUM_ITEM_FEATURES * sizeof(float);
    float* items = (float*)malloc(items_bytes);
    if (!items) { perror("malloc items"); return 1; }
    for (int j = 0; j < n_items; j++)
        for (int f = 0; f < NUM_ITEM_FEATURES; f++)
            items[j * NUM_ITEM_FEATURES + f] =
                randf_range(&rng, FEAT_MIN[NUM_USER_FEATURES + f], FEAT_MAX[NUM_USER_FEATURES + f]);
    printf("Generated %.1f MB of random item features\n", items_bytes / (1024.0 * 1024.0));

    /* Create context (compiles Metal shader) */
    double t0 = get_time_ms();
    ScoringContext* ctx = scoring_context_create(n_items);
    if (!ctx) { fprintf(stderr, "Failed to create Metal context\n"); return 1; }
    double compile_ms = get_time_ms() - t0;
    printf("Metal shader compile: %.0f ms\n", compile_ms);

    /* Load items (memcpy to unified memory — no PCIe on Apple Silicon) */
    t0 = get_time_ms();
    if (load_items(ctx, items, n_items) != 0) return 1;
    double load_ms = get_time_ms() - t0;
    printf("Item load (memcpy):   %10.2f ms  (%.1f GB/s)\n\n",
           load_ms, items_bytes / load_ms / 1e6);

    float* scores   = (float*)malloc((size_t)n_items * sizeof(float));
    int*   topk_idx = (int*)malloc(topk * sizeof(int));
    float* topk_sc  = (float*)malloc(topk * sizeof(float));
    if (!scores || !topk_idx || !topk_sc) { perror("malloc"); return 1; }

    /* Warmup */
    printf("Warming up (%d runs)...\n", warmup);
    for (int w = 0; w < warmup; w++) {
        score_user(ctx, user, scores);
        score_user_topk(ctx, user, topk, topk_idx, topk_sc);
    }

    /* Score verification */
    score_user(ctx, user, scores);
    float s_min = scores[0], s_max = scores[0];
    double s_sum = 0.0;
    for (int i = 0; i < n_items; i++) {
        if (scores[i] < s_min) s_min = scores[i];
        if (scores[i] > s_max) s_max = scores[i];
        s_sum += scores[i];
    }
    printf("--- Score Verification ---\n");
    printf("  Min score:  %.8f\n", s_min);
    printf("  Max score:  %.8f\n", s_max);
    printf("  Mean score: %.8f\n", s_sum / n_items);
    printf("  Score[0]:   %.8f\n", scores[0]);
    printf("  Score[N/2]: %.8f\n\n", scores[n_items/2]);

    /* Benchmark: full scoring */
    t0 = get_time_ms();
    for (int it = 0; it < iterations; it++)
        score_user(ctx, user, scores);
    double full_ms = (get_time_ms() - t0) / iterations;

    /* Benchmark: top-K */
    t0 = get_time_ms();
    for (int it = 0; it < iterations; it++)
        score_user_topk(ctx, user, topk, topk_idx, topk_sc);
    double topk_ms = (get_time_ms() - t0) / iterations;

    printf("\n--- Results ---\n");
    printf("  score_user():      %10.2f ms  (%8.2f M items/sec)\n",
           full_ms, n_items / full_ms / 1000.0);
    printf("  score_user_topk(): %10.2f ms  (%8.2f M items/sec)\n",
           topk_ms, n_items / topk_ms / 1000.0);
    printf("  Note: unified memory — no PCIe transfers on Apple Silicon\n");

    /* Top-K scaling */
    int k_values[] = {10, 50, 100, 500, 1000};
    int n_kv = 5;
    printf("\n--- Top-K Scaling ---\n");
    for (int ki = 0; ki < n_kv; ki++) {
        int k = k_values[ki];
        if (k > n_items || k > 1024) break;
        int*   ki_idx = (int*)malloc(k * sizeof(int));
        float* ki_sc  = (float*)malloc(k * sizeof(float));
        score_user_topk(ctx, user, k, ki_idx, ki_sc); /* warmup */
        t0 = get_time_ms();
        for (int it = 0; it < iterations; it++)
            score_user_topk(ctx, user, k, ki_idx, ki_sc);
        double k_ms = (get_time_ms() - t0) / iterations;
        printf("  K=%-5d  %10.2f ms  (%8.2f M items/sec)\n",
               k, k_ms, n_items / k_ms / 1000.0);
        free(ki_idx); free(ki_sc);
    }

    /* Top-K preview */
    printf("\nTop-%d preview:\n", topk < 10 ? topk : 10);
    int show = topk < 10 ? topk : 10;
    for (int i = 0; i < show; i++)
        printf("  rank[%d] = item[%d]  score=%.8f\n", i, topk_idx[i], topk_sc[i]);
    if (topk > 10) printf("  ... (%d more)\n", topk - 10);

    scoring_context_destroy(ctx);
    free(items); free(scores);
    free(topk_idx); free(topk_sc);
    printf("\nDone.\n");
    } /* @autoreleasepool */
    return 0;
}