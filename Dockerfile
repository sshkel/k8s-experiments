FROM fedora:27

# H2O version
ARG VERSION=3.20.0.10

# Install Java
RUN dnf install -y java-1.8.0-openjdk && \
    dnf install -y zip && \
    dnf install -y iputils && \
    dnf install -y bind-utils && \
    dnf install -y hostname && \
    dnf clean all
# Unpack H2O
COPY h2o-${VERSION}.zip /
RUN unzip h2o-${VERSION}.zip && rm -f h2o-${VERSION}.zip

RUN cd /h2o-${VERSION} && \
    pip3 install $(find . -name "*.whl")

# Install start script
COPY start /start
RUN chmod +x /start

# Run /start script
WORKDIR h2o-${VERSION}
CMD [ "/start" ]
