#include "scoring_split_core.h"
#include <time.h>

static const float FEAT_MIN[39] = {
    0.00f, 0.60f, 0.80f, 3.40f, 0.00f, 0.00f, 0.00f, 0.00f,
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
    0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
};
static const float FEAT_MAX[39] = {
    14.20f, 17.40f, 23.20f, 28.60f, 884.76f, 914.58f, 720.08f, 1267.48f,
    31769.42f, 32702.02f, 34583.37f, 20991.86f, 31692.69f, 22.60f, 34.60f, 46.40f,
    58.40f, 69.20f, 414.80f, 500.00f, 540.80f, 654.80f, 705.20f, 480.80f,
    642.80f, 776.00f, 800.00f, 849.20f, 2034.80f, 2050.40f, 1934.00f, 2219.60f,
    2645.60f, 2444.00f, 1848.80f, 1872.80f, 3351.20f, 3045.20f, 2724.80f,
};

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* RNG: uniform in [lo, hi] */
static float randf_range(unsigned int *state, float lo, float hi) {
    *state = *state * 1103515245u + 12345u;
    float t = (float)(*state >> 16) / 65535.0f;
    return lo + t * (hi - lo);
}

int main(int argc, char** argv) {
    int n_items    = 100000;
    int topk       = 100;
    int warmup     = 2;
    int iterations = 5;
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
                "\n  Defaults: -n 100000 -k 100 -w 2 -i 5 -s 42\n", argv[0]);
            return (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) ? 0 : 1;
        }
    }

    printf("=== CPU Split-Feature Scoring Benchmark ===\n");
    printf("  Items:         %d\n", n_items);
    printf("  Top-K:         %d\n", topk);
    printf("  User features: %d\n", NUM_USER_FEATURES);
    printf("  Item features: %d\n", NUM_ITEM_FEATURES);
    printf("  Trees:         %d\n", NUM_TREES);
    printf("  Warmup:        %d\n", warmup);
    printf("  Iterations:    %d\n", iterations);
    printf("  Seed:          %u\n\n", seed);

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
            items[j * NUM_ITEM_FEATURES + f] = randf_range(&rng, FEAT_MIN[NUM_USER_FEATURES + f], FEAT_MAX[NUM_USER_FEATURES + f]);
    printf("Generated %.1f MB of random item features\n", items_bytes / (1024.0 * 1024.0));

    float* scores = (float*)malloc((size_t)n_items * sizeof(float));
    if (!scores) { perror("malloc scores"); return 1; }

    /* ---- Warmup ---- */
    printf("Warming up (%d runs)...\n", warmup);
    for (int w = 0; w < warmup; w++)
        score_user_cpu(user, items, scores, n_items);

    /* ---- Score verification ---- */
    score_user_cpu(user, items, scores, n_items);
    float s_min = scores[0], s_max = scores[0];
    double s_sum = 0.0;
    int unique_count = 0;
    for (int i = 0; i < n_items; i++) {
        if (scores[i] < s_min) s_min = scores[i];
        if (scores[i] > s_max) s_max = scores[i];
        s_sum += scores[i];
    }
    /* Count unique scores (sample first 100K) */
    {
        int check_n = n_items < 100000 ? n_items : 100000;
        float* sample = (float*)malloc(check_n * sizeof(float));
        int stride = (n_items + check_n - 1) / check_n;
        for (int i = 0; i < check_n; i++) {
            int idx = i * stride;
            if (idx >= n_items) idx = n_items - 1;
            sample[i] = scores[idx];
        }
        /* simple unique count via sort */
        for (int i = 0; i < check_n - 1; i++)
            for (int j = i + 1; j < check_n && j < i + 50; j++)
                if (sample[j] < sample[i]) { float t = sample[i]; sample[i] = sample[j]; sample[j] = t; }
        unique_count = 1;
        for (int i = 1; i < check_n; i++)
            if (sample[i] != sample[i-1]) unique_count++;
        free(sample);
    }
    printf("--- Score Verification ---\n");
    printf("  Min score:     %.8f\n", s_min);
    printf("  Max score:     %.8f\n", s_max);
    printf("  Mean score:    %.8f\n", s_sum / n_items);
    printf("  Unique scores: %d (sampled)\n", unique_count);
    printf("  Score[0]:      %.8f\n", scores[0]);
    printf("  Score[N/4]:    %.8f\n", scores[n_items/4]);
    printf("  Score[N/2]:    %.8f\n", scores[n_items/2]);
    printf("  Score[3N/4]:   %.8f\n", scores[3*n_items/4]);
    printf("  Score[N-1]:    %.8f\n\n", scores[n_items-1]);

    /* ---- Benchmark: full scoring ---- */
    double t0 = get_time_ms();
    for (int it = 0; it < iterations; it++)
        score_user_cpu(user, items, scores, n_items);
    double full_ms = (get_time_ms() - t0) / iterations;

    /* ---- Benchmark: score + top-K (qsort) ---- */
    ScoredItem* ranked = (ScoredItem*)malloc(n_items * sizeof(ScoredItem));
    int*   topk_idx = (int*)malloc(topk * sizeof(int));
    float* topk_sc  = (float*)malloc(topk * sizeof(float));
    if (!ranked || !topk_idx || !topk_sc) { perror("malloc"); return 1; }

    t0 = get_time_ms();
    for (int it = 0; it < iterations; it++) {
        score_user_cpu(user, items, scores, n_items);
        for (int i = 0; i < n_items; i++) {
            ranked[i].score = scores[i];
            ranked[i].index = i;
        }
        qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);
        int rk = topk < n_items ? topk : n_items;
        for (int i = 0; i < rk; i++) {
            topk_idx[i] = ranked[i].index;
            topk_sc[i]  = ranked[i].score;
        }
    }
    double topk_ms = (get_time_ms() - t0) / iterations;

    /* ---- Results ---- */
    printf("\n--- Results ---\n");
    printf("  Full scoring:      %10.2f ms  (%8.2f K items/sec)\n",
           full_ms, n_items / full_ms);
    printf("  Score + top-%-4d:  %10.2f ms  (%8.2f K items/sec)\n",
           topk, topk_ms, n_items / topk_ms);
    printf("  Top-K overhead:    %10.2f ms  (%.1f%% of scoring)\n",
           topk_ms - full_ms,
           full_ms > 0 ? (topk_ms - full_ms) / full_ms * 100.0 : 0.0);

    /* ---- Top-K scaling ---- */
    int k_values[] = {10, 50, 100, 500, 1000};
    int n_kv = 5;
    printf("\n--- Top-K Scaling (score + qsort) ---\n");
    for (int ki = 0; ki < n_kv; ki++) {
        int k = k_values[ki];
        if (k > n_items) break;
        int*   ki_idx = (int*)malloc(k * sizeof(int));
        float* ki_sc  = (float*)malloc(k * sizeof(float));
        if (!ki_idx || !ki_sc) continue;
        t0 = get_time_ms();
        for (int it = 0; it < iterations; it++) {
            score_user_cpu(user, items, scores, n_items);
            for (int j = 0; j < n_items; j++) {
                ranked[j].score = scores[j];
                ranked[j].index = j;
            }
            qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);
            int rk = k < n_items ? k : n_items;
            for (int j = 0; j < rk; j++) {
                ki_idx[j] = ranked[j].index;
                ki_sc[j]  = ranked[j].score;
            }
        }
        double k_ms = (get_time_ms() - t0) / iterations;
        printf("  K=%-5d  %10.2f ms  (%8.2f K items/sec)\n",
               k, k_ms, n_items / k_ms);
        free(ki_idx); free(ki_sc);
    }

    /* ---- Top-K preview ---- */
    printf("\nTop-%d preview:\n", topk < 10 ? topk : 10);
    int show = topk < 10 ? topk : 10;
    for (int i = 0; i < show; i++)
        printf("  rank[%d] = item[%d]  score=%.8f\n", i, topk_idx[i], topk_sc[i]);
    if (topk > 10) printf("  ... (%d more)\n", topk - 10);

    free(items); free(scores); free(ranked);
    free(topk_idx); free(topk_sc);
    printf("\nDone.\n");
    return 0;
}