#include "scoring_split_core.h"

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
    score_user_cpu(user, items, scores, n_items);

    if (topk > 0) {
        /* Top-K: sort all scores, output top K */
        ScoredItem* ranked = (ScoredItem*)malloc(n_items * sizeof(ScoredItem));
        for (int i = 0; i < n_items; i++) {
            ranked[i].score = scores[i];
            ranked[i].index = i;
        }
        qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);
        int result_k = topk < n_items ? topk : n_items;

        if (output_path) {
            FILE* fout = fopen(output_path, "wb");
            if (!fout) { perror("fopen output"); return 1; }
            for (int i = 0; i < result_k; i++) {
                int idx = ranked[i].index;
                fwrite(&idx, sizeof(int), 1, fout);
            }
            for (int i = result_k; i < topk; i++) {
                int neg = -1;
                fwrite(&neg, sizeof(int), 1, fout);
            }
            for (int i = 0; i < result_k; i++) {
                float s = ranked[i].score;
                fwrite(&s, sizeof(float), 1, fout);
            }
            for (int i = result_k; i < topk; i++) {
                float neg = -1e30f;
                fwrite(&neg, sizeof(float), 1, fout);
            }
            fclose(fout);
            fprintf(stderr, "Wrote top-%d results to %s\n", topk, output_path);
        } else {
            for (int i = 0; i < result_k; i++)
                printf("rank[%d] = item[%d]  score=%.8f\n", i, ranked[i].index, ranked[i].score);
        }
        free(ranked);
    } else {
        /* Full scoring mode */
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
