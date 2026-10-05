# Dependencies only: HTCondor transfers the current test.py and src/ per job.
FROM python:3.11-slim-bookworm

ARG TARGETARCH
RUN if [ "$TARGETARCH" != "amd64" ]; then \
      printf 'This cluster image requires AMD64. Rebuild with: docker buildx build --platform linux/amd64 --load -t IMAGE .\n' >&2; \
      exit 1; \
    fi

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/sharpness/requirements.txt
# Explicit CUDA wheels: do not accidentally build a CPU-only image.
RUN python -m pip install --no-cache-dir torch==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --no-cache-dir -r /opt/sharpness/requirements.txt \
    && python -m pip check \
    && python -c 'import torch, numpy, wandb; from torch.func import functional_call; from torch.nn.attention import SDPBackend, sdpa_kernel; assert torch.version.cuda == "12.6"; print(torch.__version__, numpy.__version__, wandb.__version__)'

# HTCondor supplies the runtime user and working directory.
CMD ["/bin/bash"]
