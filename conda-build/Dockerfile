FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x ./miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV CONDA_PREFIX=/opt/conda
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN git -C ~/ clone https://github.com/apasarkar/masknmf.git \
    && /opt/conda/bin/conda install conda-build \
    && chmod +x ~/masknmf/conda-build/conda-build.sh

WORKDIR /root/masknmf/conda-build

# TODO: possible to securely configure anaconda-client to automate login?
# Build Conda Environments
ENTRYPOINT ["bash", "-c", "~/masknmf/conda-build/conda-build.sh"]