#include "scoring_split_core.cuh"

/* ==== Standalone Entry Point ==== */

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <user.bin> <items.bin> [output.bin] [-k topK]\n"
            "\n"
            "  user.bin:   binary float32 [%d] (one user feature vector)\n"
            "  items.bin:  binary float32 [N x %d] (item features, row-major)\n"
            "  output.bin: binary output (omit for stdout)\n"
            "  -k topK:    return only the top K results (ranked by score)\n",
            argv[0], NUM_USER_FEATURES, NUM_ITEM_FEATURES);
        return 1;
    }

    /* Parse arguments */
    const char* user_path = argv[1];
    const char* items_path = argv[2];
    const char* output_path = NULL;
    int topk = 0;

    for (int a = 3; a < argc; a++) {
        if (strcmp(argv[a], "-k") == 0 && a + 1 < argc) {
            topk = atoi(argv[++a]);
        } else if (!output_path) {
            output_path = argv[a];
        }
    }

    /* Read user features */
    FILE* fu = fopen(user_path, "rb");
    if (!fu) { perror("fopen user"); return 1; }
    float user[NUM_USER_FEATURES];
    if (fread(user, sizeof(float), NUM_USER_FEATURES, fu) != NUM_USER_FEATURES) {
        fprintf(stderr, "Failed to read %d user features\n", NUM_USER_FEATURES);
        fclose(fu);
        return 1;
    }
    fclose(fu);

    /* Read item features */
    FILE* fi = fopen(items_path, "rb");
    if (!fi) { perror("fopen items"); return 1; }
    fseek(fi, 0, SEEK_END);
    long fsize = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    int n_items = (int)(fsize / ((long)NUM_ITEM_FEATURES * (long)sizeof(float)));
    if (n_items <= 0) {
        fprintf(stderr, "Invalid items file\n");
        fclose(fi);
        return 1;
    }

    if (topk > 0) {
        fprintf(stderr, "Top-%d ranking of %d items (%d user + %d item features)...\n",
                topk, n_items, NUM_USER_FEATURES, NUM_ITEM_FEATURES);
    } else {
        fprintf(stderr, "Scoring %d items against 1 user (%d user + %d item features)...\n",
                n_items, NUM_USER_FEATURES, NUM_ITEM_FEATURES);
    }

    float* items = (float*)malloc((size_t)fsize);
    if (!items) { perror("malloc items"); return 1; }
    fread(items, sizeof(float), (size_t)n_items * NUM_ITEM_FEATURES, fi);
    fclose(fi);

    float* scores = (float*)malloc((size_t)n_items * sizeof(float));
    if (!scores) { perror("malloc scores"); return 1; }

    /* Score */
    ScoringContext* ctx = scoring_context_create(n_items);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    if (load_items(ctx, items, n_items) != 0) return 1;

    if (topk > 0) {
        /* Top-K mode: fused scoring + selection */
        int* topk_indices = (int*)malloc(topk * sizeof(int));
        float* topk_scores = (float*)malloc(topk * sizeof(float));
        if (!topk_indices || !topk_scores) { perror("malloc topk"); return 1; }
        if (score_user_topk(ctx, user, topk, topk_indices, topk_scores) != 0) return 1;
        scoring_context_destroy(ctx);

        if (output_path) {
            FILE* fout = fopen(output_path, "wb");
            if (!fout) { perror("fopen output"); return 1; }
            fwrite(topk_indices, sizeof(int), topk, fout);
            fwrite(topk_scores, sizeof(float), topk, fout);
            fclose(fout);
            fprintf(stderr, "Wrote top-%d results to %s\n", topk, output_path);
        } else {
            for (int i = 0; i < topk; i++)
                printf("rank[%d] = item[%d]  score=%.8f\n", i, topk_indices[i], topk_scores[i]);
        }
        free(topk_indices);
        free(topk_scores);
    } else {
        /* Full scoring mode */
        if (score_user(ctx, user, scores) != 0) return 1;
        scoring_context_destroy(ctx);

        if (output_path) {
            FILE* fout = fopen(output_path, "wb");
            if (!fout) { perror("fopen output"); return 1; }
            fwrite(scores, sizeof(float), (size_t)n_items, fout);
            fclose(fout);
            fprintf(stderr, "Wrote %d scores to %s\n", n_items, output_path);
        } else {
            int limit = n_items < 20 ? n_items : 20;
            for (int i = 0; i < limit; i++)
                printf("item[%d] = %.8f\n", i, scores[i]);
            if (n_items > 20)
                printf("... (%d more items)\n", n_items - 20);
        }
    }

    free(items);
    free(scores);
    return 0;
}
