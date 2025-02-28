FROM ghcr.io/ggerganov/llama.cpp:light

# Set the working directory to your project root
WORKDIR /workspaces/interview_analyzer_ensemble

# Install system packages and build tools
RUN apt-get update && apt-get install -y \
     ffmpeg \
     sox \
     libsndfile1 \
     git \
     build-essential \
     cmake \
     curl \
     python3.10 \
     python3.10-venv \
     python3.10-dev \
     python3-pip \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
  && ln -sf /usr/bin/pip3 /usr/bin/pip \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set LD_LIBRARY_PATH so the dynamic linker can find shared libraries
ENV LD_LIBRARY_PATH="/usr/ctransformers/lib/local:/usr/lib:/app"

# Build and install ctransformers from source for ARM (v0.2.27)
RUN apt-get update && apt-get install -y git cmake build-essential && \
    git clone --recurse-submodules https://github.com/marella/ctransformers.git /tmp/ctransformers && \
    cd /tmp/ctransformers && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
    make -j$(nproc) && \
    make install && ldconfig && \
    cd /tmp/ctransformers && pip install . && \
    cd / && rm -rf /tmp/ctransformers

# Copy the project files into the container
COPY . .

# Install Python dependencies (ensure ctransformers is not included in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Command to run when the container starts
CMD ["python", "src/main.py"]
