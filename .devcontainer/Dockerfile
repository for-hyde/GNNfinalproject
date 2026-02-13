FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Install git and other essential tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a developer user (optional, but good practice)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    matplotlib \
    pandas \
    scikit-learn \
    tensorboard \
    tqdm \
    seaborn

# Set working directory
WORKDIR /workspace

# Switch to developer user
USER $USERNAME

# Expose Jupyter port
EXPOSE 8888