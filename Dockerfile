FROM tensorflow/serving

WORKDIR /tmp
RUN apt update && \
    apt install -y curl && \
    curl -OL https://github.com/jeongukjae/multilingual-bert-korean-hate-speech/releases/download/0/model.tar.gz && \
    tar -zxvf model.tar.gz && \
    rm -rf /models/mode && \
    cp -rp hate-speech-model /models/model && \
    rm -rf /tmp

EXPOSE 8501
