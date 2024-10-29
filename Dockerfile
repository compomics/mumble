FROM python:3.12

ADD pyproject.toml /mumble/pyproject.toml
ADD LICENSE /mumble/LICENSE
ADD README.md /mumble/README.md
ADD mumble /mumble/mumble

RUN apt-get update \
    && apt-get install -y procps \
    && pip install /mumble --only-binary :all:

ENTRYPOINT [""]
