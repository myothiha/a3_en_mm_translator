FROM python:3.11.5-bookworm

RUN pip install --upgrade pip

WORKDIR /root/source_code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm

# CMD tail -f /dev/null