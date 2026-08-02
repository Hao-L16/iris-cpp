// load_gguf.cpp — smallest possible check: open an IRIS .gguf, list its
// metadata keys and every tensor's name + shape + type. No compute graph yet.
// If this prints all 16 actor-critic tensors with the shapes we expect, the
// export + C++ read path is proven and we can start building the graph.

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cinttypes>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s path/to/model.gguf\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    // gguf_init_from_file also allocates a ggml_context holding the tensor
    // metadata (and data, since no_alloc=false). We only read here.
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/&ctx };

    struct gguf_context* gguf = gguf_init_from_file(path, params);
    if (!gguf) {
        printf("FAILED to open %s\n", path);
        return 1;
    }

    // --- metadata ---
    const int64_t n_kv = gguf_get_n_kv(gguf);
    printf("=== %s ===\n", path);
    printf("metadata: %" PRId64 " keys\n", n_kv);
    for (int64_t i = 0; i < n_kv; i++) {
        const char* key = gguf_get_key(gguf, i);
        enum gguf_type t = gguf_get_kv_type(gguf, i);
        printf("  [%02" PRId64 "] %-24s ", i, key);
        if (t == GGUF_TYPE_UINT32) {
            printf("u32 = %u\n", gguf_get_val_u32(gguf, i));
        } else if (t == GGUF_TYPE_FLOAT32) {
            printf("f32 = %g\n", gguf_get_val_f32(gguf, i));
        } else if (t == GGUF_TYPE_STRING) {
            printf("str = %s\n", gguf_get_val_str(gguf, i));
        } else {
            printf("(type %d)\n", (int)t);
        }
    }

    // --- tensors ---
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    printf("\ntensors: %" PRId64 "\n", n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx, name);
        printf("  %-45s ne=[%5" PRId64 " %5" PRId64 " %5" PRId64 " %5" PRId64 "]  %s\n",
               name, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
               ggml_type_name(t->type));
    }

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}
