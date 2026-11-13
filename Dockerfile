# SPDX-License-Identifier: MIT
# Both arguments name freshly built stages from build.sh, not production images.
ARG PUBLIC_BUILDER
ARG PUBLIC_RUNTIME
FROM ${PUBLIC_BUILDER} AS build
COPY patches /recipe-patches
RUN cd /tensorfold && for p in /recipe-patches/*.patch; do git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig fmt zig/src/core/lanes/engine.zig zig/src/families/flashnext/cuda_head8.zig zig/src/families/flashnext/cuda_prompt.zig zig/src/families/flashnext/cuda_batch_head.zig zig/src/families/flashnext/cuda_batch_state.zig zig/src/families/flashnext/cuda_lanes.zig && \
    sha256sum -c /recipe-patches/production-zig-sources.sha256 >/dev/null && \
    mkdir -p /trial && cp head8/* /trial/ && \
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_121 --fatbin /trial/head8_complete.cu -o zig/src/families/flashnext/head8.fatbin && \
    /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold -j2 native
FROM ${PUBLIC_RUNTIME}
COPY --from=build /opt/tensorfold /opt/tensorfold
# Preserve the complete source checkout and new CUDA build inputs for future audits.
COPY --from=build /tensorfold /tensorfold
COPY NOTICE CREDITS /opt/tensorfold/share/doc/recipe/
LABEL org.opencontainers.image.source="https://github.com/sojufx/sojufx-Qwen3.8-Flash-Next" recipe.miaai.commit="20e15e41f2f514a0d287710f4665ee1b5009ba41"
