# this container inherits its entry point from the rstudio container
# which is build on r-ver

# build for multiple platforms:
# docker buildx build --platform linux/amd64,linux/arm64 -t npcooley/container_name:0.0.1

# use: docker run -it --rm npcooley/container_name:x.y.z bash
# this is set up on r-studio, so interactive requires an entry point override
# spin it up interactively for testing and things

# other args:
# inherit gpu devices from the host:
# --gpus all

# inherit the icd files from the host:
# -v /etc/OpenCL/vendors/:/etc/OpenCL/vendors/

# log in to the instance
# http://localhost:8787

FROM rocker/rstudio:4.6.1

# system dependencies
RUN apt-get update && apt-get install -y \
      bash-completion \
      opencl-headers \
      ocl-icd-libopencl1 \
      ocl-icd-opencl-dev \
      clinfo \
      build-essential \
      software-properties-common \
      nano \
      git \
      curl \
      tree \
      libgomp1 \
      libuv1 \
      zlib1g-dev \
      texlive-base \
      texlive-latex-base \
      texlive-fonts-recommended && \
      apt-get -y autoclean && \
      rm -rf /var/lib/apt/lists/*

# grab CUDA directly from NVIDIA avoid pythonic installations in an effort to
# limit the already messy overhead
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
      cuda-toolkit-12-6 \
      libcudnn9-cuda-12 \
      libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# an attempt at layering install steps ...
RUN Rscript -e "install.packages(c('gitcreds', 'knitr', 'rmarkdown'))"

# packages that are usecase specific
RUN Rscript -e "install.packages('OpenCL', type = 'source')"

# ENV CUML_PREFIX=/opt/conda/envs/rapids
# RUN Rscript -e "install.packages('cuda.ml', type = 'source', repos = 'https://p3m.dev/cran/latest')"

ENV CUDA=12.6
RUN Rscript -e "install.packages('torch')"
RUN Rscript -e "torch::install_torch()"

# RUN echo "/opt/conda/envs/rapids/lib" > /etc/ld.so.conf.d/rapids-cuda.conf && \
#     ldconfig

RUN Rscript -e "install.packages('hfhub')"

RUN Rscript -e "install.packages('BiocManager')"
RUN Rscript -e "BiocManager::install('Biostrings')"

# -p creates intermediate directories if they do not exist
RUN mkdir -p /home/rstudio/.config/rstudio
COPY rstudio-prefs.json /home/rstudio/.config/rstudio/rstudio-prefs.json

