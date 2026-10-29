FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385
RUN python -m pip install --no-cache-dir \
    transformers==4.57.6 numpy==2.4.6 pillow==12.3.0 openvino==2026.2.1 \
    sentencepiece==0.2.1 protobuf==7.36.2
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
WORKDIR /workspace
COPY train.py /workspace/train.py
ENTRYPOINT ["python", "/workspace/train.py"]
