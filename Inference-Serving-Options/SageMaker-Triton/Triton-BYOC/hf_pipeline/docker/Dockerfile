ARG BASE_IMAGE

FROM $BASE_IMAGE

#Install any additional libraries
RUN echo "Adding conda package to Docker image"
RUN mkdir -p /home/condpackenv/

# Copy conda env
COPY stablediff_env.tar.gz /home/condpackenv/stablediff_env.tar.gz