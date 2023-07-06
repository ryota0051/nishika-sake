FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG WORK_DIR=/work

WORKDIR ${WORK_DIR}

RUN apt-get update && apt-get install curl -y

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION="1.5.0"

RUN pip install --upgrade pip && pip uninstall -y virtualenv

RUN curl -sSSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml* ./poetry.lock* ./

RUN poetry install

ENV PYTHONPATH ${WORK_DIR}:${PYTHONPATH}

CMD [ "jupyter-lab", "--ip", "0.0.0.0", "--allow-root", "--NotebookApp.token=''" ]
