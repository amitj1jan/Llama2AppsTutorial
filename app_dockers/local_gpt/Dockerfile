# syntax=docker/dockerfile:1

# using this as base image to keep the application image size as low as possible
FROM python:3.10-alpine

# istalls libraries build-base, gcc and pip libraries
RUN apk update && apk add --virtual build-dependencies build-base gcc py3-pip

# this is our working directory on container
WORKDIR /app

# copies the application code, requirements file and welcome page config for app
COPY app.py requirements.txt chainlit.md ./

# installs the application dependencies using pip
RUN pip install -r requirements.txt

# exposes the port, where local_gpt app can be accessed
EXPOSE 8001

# this command runs the starts the genAI app on container
CMD ["chainlit",  "run", "app.py", "--port", "8001"]
