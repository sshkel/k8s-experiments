FROM appropriate/curl
RUN curl -sSL -o /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/v1.12.0/bin/linux/amd64/kubectl && \
    curl -sSL -o /usr/local/bin/argo https://github.com/argoproj/argo/releases/download/v2.2.1/argo-linux-amd64 && \
    chmod +x /usr/local/bin/argo && \
    chmod +x /usr/local/bin/kubectl
