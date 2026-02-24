FROM apache/airflow:3.0.6-python3.11

USER root

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk g++ && \
    chmod a+x /usr/bin/g++ /usr/bin/gcc && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt