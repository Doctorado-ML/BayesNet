FROM mcr.microsoft.com/devcontainers/cpp:ubuntu22.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.29.3"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
    chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends wget software-properties-common libdatetime-perl libcapture-tiny-perl libdatetime-format-dateparse-perl libgd-perl

# Add PPA for GCC 13
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update

# Install GCC 13.1
RUN apt-get install -y gcc-13 g++-13 doxygen

# Install lcov 2.1
RUN wget --quiet https://github.com/linux-test-project/lcov/releases/download/v2.1/lcov-2.1.tar.gz && \
    tar -xvf lcov-2.1.tar.gz && \
    cd lcov-2.1 && \
    make install
RUN rm lcov-2.1.tar.gz
RUN rm -fr lcov-2.1

# Install Miniconda
RUN mkdir -p /opt/conda 
# RUN wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" -O /opt/conda/miniconda.sh && \
RUN wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /opt/conda/miniconda.sh && \
    bash /opt/conda/miniconda.sh -b -p /opt/miniconda

# Add conda to PATH
ENV PATH=/opt/miniconda/bin:$PATH

# add CXX and CC to the environment with gcc 13
ENV CXX=/usr/bin/g++-13
ENV CC=/usr/bin/gcc-13

# link the last gcov version
RUN rm /usr/bin/gcov
RUN ln -s /usr/bin/gcov-13 /usr/bin/gcov

# change ownership of /opt/miniconda to vscode user
RUN chown -R vscode:vscode /opt/miniconda

USER vscode
RUN conda init
RUN conda install -y -c conda-forge yaml pytorch