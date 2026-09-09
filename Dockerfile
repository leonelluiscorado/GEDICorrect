# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS hancock-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates csh g++ gcc git libgdal-dev libgeotiff-dev \
        libgsl-dev libhdf5-dev libtiff-dev make \
    && rm -rf /var/lib/apt/lists/*

COPY install_hancock_tools.bash /tmp/install_hancock_tools.bash
RUN GEDICORRECT_PREFIX=/opt/hancock-tools \
    GEDICORRECT_SKIP_SYSTEM_PACKAGES=1 \
    GEDICORRECT_PERSIST_ENV=0 \
    bash /tmp/install_hancock_tools.bash


FROM python:3.12-slim-bookworm AS runtime

ARG GEDICORRECT_UID=10001
ARG GEDICORRECT_GID=10001

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        csh libgdal32 libgeotiff5 libgsl27 libhdf5-103-1 libtiff6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=hancock-builder /opt/hancock-tools /opt/hancock-tools
ENV PATH="/opt/hancock-tools/bin:${PATH}" \
    GEDICORRECT_CONTAINER=1 \
    GEDICORRECT_JOB_STATE=/data/output/.gedicorrect/job.json \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN for program in gediRat gediMetric; do \
        if ldd "/opt/hancock-tools/bin/${program}" | grep -q "not found"; then \
            echo "Missing runtime library for ${program}" >&2; \
            exit 1; \
        fi; \
    done

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[ui,laz,raster]" \
    && gedicorrect check

RUN groupadd --gid "${GEDICORRECT_GID}" gedicorrect \
    && useradd --uid "${GEDICORRECT_UID}" --gid "${GEDICORRECT_GID}" --create-home gedicorrect \
    && mkdir -p /data/als /data/input /data/output \
    && chown -R gedicorrect:gedicorrect /data

USER gedicorrect
WORKDIR /data
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

ENTRYPOINT ["gedicorrect"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8501"]
