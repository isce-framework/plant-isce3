FROM --platform=linux/amd64 oraclelinux:8

LABEL author="Gustavo H. X. Shiroma" \
    description="PLAnT-ISCE3" \
    version="1.0.2"

RUN yum -y update &&\
    yum -y install curl &&\
    adduser plant_isce3_user

RUN mkdir -p /home/plant_isce3_user/plant-isce3

RUN chmod -R 755 /home/plant_isce3_user &&\
    chown -R plant_isce3_user:plant_isce3_user /home/plant_isce3_user/plant-isce3

USER plant_isce3_user

ENV CONDA_PREFIX=/home/plant_isce3_user/miniforge3

# install miniforge
WORKDIR /home/plant_isce3_user
RUN curl -sSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o miniforge.sh &&\
    bash miniforge.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniforge.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}

COPY ../requirements.txt .

RUN ${CONDA_PREFIX}/bin/conda init bash && \
    conda config --prepend channels conda-forge &&\
    conda config --append channels nodefaults &&\
    conda config --set channel_priority strict &&\
    conda config --set show_channel_urls True &&\
    conda update --all --yes --channel conda-forge --override-channels &&\
    conda install --yes --channel conda-forge --override-channels --file requirements.txt &&\
    conda install --yes --channel conda-forge pytest

# copy PLAnT-ISCE3 source code and set plant_isce3_user as owner
COPY --chown=plant_isce3_user:plant_isce3_user . /home/plant_isce3_user/plant-isce3

RUN python -m pip install  --no-deps /home/plant_isce3_user/plant-isce3 &&\
    echo "conda activate isce3" >> /home/plant_isce3_user/.bashrc

WORKDIR /home/plant_isce3_user/plant-isce3
