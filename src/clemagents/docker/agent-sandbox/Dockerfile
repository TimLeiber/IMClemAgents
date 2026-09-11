FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CHROME_BIN=/usr/bin/chromium
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    zsh \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    diffutils \
    patch \
    tar \
    gzip \
    unzip \
    zip \
    xz-utils \
    bzip2 \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    make \
    cmake \
    pkg-config \
    ca-certificates \
    openssl \
    gnupg \
    dnsutils \
    iputils-ping \
    netcat-openbsd \
    procps \
    psmisc \
    lsof \
    jq \
    sqlite3 \
    ripgrep \
    fd-find \
    file \
    less \
    tree \
    vim-tiny \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    chromium \
    xvfb \
    wamerican \
    wordnet \
    aspell \
    aspell-en \
    hunspell \
    hunspell-en-us \
    imagemagick \
    tesseract-ocr \
    tesseract-ocr-eng \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        requests \
        httpx \
        beautifulsoup4 \
        lxml \
        html5lib \
        trafilatura \
        readability-lxml \
        pandas \
        numpy \
        scipy \
        sympy \
        networkx \
        python-dateutil \
        rapidfuzz \
        duckdb \
        pyarrow \
        regex \
        tqdm \
        pyyaml \
        matplotlib \
        pillow \
        opencv-python-headless \
        scikit-image \
        imageio \
        imagehash \
        pytesseract \
        nltk \
        wordfreq \
        wordninja

# Keep OpenClaw reproducible across image rebuilds. Version 2026.8.11 is the
# version recorded in the last known-successful OpenRouter benchmark traces. first episodes ran 2026.6.11
RUN npm install -g @anthropic-ai/claude-code @openai/codex openclaw@2026.8.1
# install the native search provider separately, pinned to the openclaw release
RUN npm install -g --omit=peer @openclaw/duckduckgo-plugin@2026.8.1 && \
    mkdir -p /opt/openclaw-plugins && \
    ln -s "$(npm root -g)/@openclaw/duckduckgo-plugin" /opt/openclaw-plugins/duckduckgo
COPY pyproject.toml README.md LICENSE /opt/clemagents/
COPY src /opt/clemagents/src

RUN pip install --no-cache-dir /opt/clemagents && \
    pip install --no-cache-dir \
        claude-agent-sdk \
        ddgs==9.14.4 \
        fastmcp \
        openai-codex \
        "hermes-agent[cli,mcp]"

RUN useradd -m -u 1000 agent && \
    mkdir -p /workspace /ms-playwright && \
    chown -R agent:agent /workspace /ms-playwright

USER agent
WORKDIR /workspace

ENV PYTHONPATH=/opt/clemagents/src

CMD ["bash"]
