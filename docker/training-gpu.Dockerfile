FROM rust:1.88-bookworm

# This image is deliberately a local Rust/Python/CUDA training runtime.  It
# does not install Node, a browser, or Playwright; those belong to UI release QA.

ARG TORCH_VERSION=2.11.0+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG NUMPY_VERSION=2.3.5

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        --break-system-packages \
        --no-cache-dir \
        --upgrade \
        pip \
        setuptools \
        wheel

RUN pip3 install \
        --break-system-packages \
        --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && pip3 install \
        --break-system-packages \
        --no-cache-dir \
        maturin==1.14.1

# Keep this after the large Torch layer so adding Python utilities does not
# force another multi-gigabyte CUDA wheel download.
RUN pip3 install \
        --break-system-packages \
        --no-cache-dir \
        "numpy==${NUMPY_VERSION}"

WORKDIR /work
CMD ["sleep", "infinity"]
