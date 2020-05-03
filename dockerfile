FROM continuumio/miniconda3

RUN mkdir /openTriage

WORKDIR /openTriage

# FOR USING FASTTEXT need build-essential gcc
RUN apt update
RUN apt install -y build-essential

# CONDA ENV SETUP
RUN conda update -n base -c defaults conda

# creating enviroment with name in order to save time with update.
RUN conda create --verbose --name openTriage python=3.7

#Activate conda enviroment
RUN echo "source activate openTriage" > ~/.bashrc
ENV PATH /opt/conda/envs/openTriage/bin:$PATH

# Copy the eniroment file
COPY lib/conda_env.yml lib/

# Update the environment with the lastest conda_env.yml
# TODO: Split this up so that only packages needed by specified framework are loaded?
RUN conda env update -f lib/conda_env.yml 

EXPOSE 5000

# Copy the current directory into the image
COPY . .