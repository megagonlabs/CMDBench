# syntax=docker/dockerfile:1
FROM ubuntu:24.04
WORKDIR /cmdbench
RUN apt update && \
    apt -y install python3 python3-pip python-is-python3 curl tzdata gnupg2 lsb-release wget nano && \
    sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg && \
    apt update && \
    apt -y install postgresql-16 postgresql-contrib-16 libpq-dev
COPY ./requirements.txt /cmdbench/requirements.txt
RUN pip3 install --break-system-packages -r requirements.txt
COPY . /cmdbench