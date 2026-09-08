//dump_types.cpp
#include "ggml.h"
#include "gguf.h"
#include <cstdio>

int main(int argc, char ** argv){
    for(int a = 1; a < argc; a++){
        struct ggml_context * ctx = nullptr;
        struct gguf_init_params p = {/*no_alloc*/ false, /*ctx*/ &ctx};
        struct gguf_context * g = gguf_init_from_file(argv[a],p);
        if(!g) { printf("open failed: %s\n", argv[a]); continue;}

        printf("\n === %s ====\n", argv[a]);
        printf("%-44s %-24s %9s %5s %5s\n", "name", "shape","MB","q32","q256");

        double total = 0, q32 = 0, q256 = 0;
        for (int64_t i =0; i< gguf_get_n_tensors(g); i++){
            const char * name = gguf_get_tensor_name(g,i);
            struct ggml_tensor * t = ggml_get_tensor(ctx,name);
            double mb = ggml_nbytes(t) / 104857.0;
            total += mb;

            bool is2d = ggml_n_dims(t) == 2;
            bool ok32  = is2d && t->ne[0] % 32 == 0;         //Q8_0
            bool ok256 = is2d && t->ne[0] % 256 == 0;        //K-quant
            if(ok32) q32 += mb;
            if(ok256) q256 += mb;

            char shape[64];
            snprintf(shape,sizeof shape, "[%lld, %lld,%lld,%lld]",
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
            printf("%-44s %-24s %9.3f %5s %5s\n",
                    name, shape, mb,ok32 ? "Y" : "-", ok256 ? "Y" : "-");
        }
        printf("\ntotal %.2f MB | Q8_0-able %.2f MB(%.1f%%) | K-quant-able %.2f MB (%.1f%%)\n",
                total,q32,100*q32/total,q256,100*q256/total);

        gguf_free(g);
        ggml_free(ctx);
    }
    return 0;
}
