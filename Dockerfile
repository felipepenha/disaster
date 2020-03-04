# base-image declaration & credentials
FROM python:3.6-slim
## For PySpark projects use this image below instead:
# FROM neowaylabs/docker-spark:2.3.1-hadoop-2.9.1

# Show Python version on build step
RUN python -V

# Build application
ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# Install 'h2o' dependencies
RUN pip install --upgrade requests
RUN pip install --upgrade tabulate
RUN pip install --upgrade colorama>=0.3.8
RUN pip install --upgrade future

# Install requirements
ADD requirements.txt .
RUN pip --disable-pip-version-check install -r requirements.txt

COPY . ${APP_DIR}
RUN pip --disable-pip-version-check install -e .
RUN chmod -R a+w ${APP_DIR}
ENTRYPOINT ["disaster"]

RUN pip install nltk==3.4.5
RUN python -m nltk.downloader -d /nltk_data punkt