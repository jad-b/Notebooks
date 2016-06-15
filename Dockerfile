# https://github.com/jupyter/docker-stacks/tree/master/all-spark-notebook
FROM jupyter/all-spark-notebook

# Install custom pip packages
RUN pip install -U pip && pip install -U \
    beautifulsoup4>=4.4.1 \
    mpld3

# Install graphlab
ENV GRAPHLAB_EMAIL j.american.db@gmail.com
ARG GRAPHLAB_LICENSE
ARG GRAPHLAB_VERSION=1.9
ENV PIP2 /opt/conda/envs/python2/bin/pip
RUN $PIP2 install --upgrade \
    https://get.dato.com/GraphLab-Create/$GRAPHLAB_VERSION/$GRAPHLAB_EMAIL/$GRAPHLAB_LICENSE/GraphLab-Create-License.tar.gz
# Install GPU-acclerated graphlab
RUN $PIP2 install --upgrade \
    http://static.dato.com/files/graphlab-create-gpu/graphlab-create-$GRAPHLAB_VERSION.gpu.tar.gz

# Install mlsl
USER root
COPY mlsl /src/mlsl
RUN pip install -e /src/mlsl['test']
# Not sure how well it behaves in Python2, guess we'll find out!
RUN $PIP2 install -e /src/mlsl['test']

# TODO Remove once IPython version >=4.2.1
#   http://stackoverflow.com/questions/37232446/ipython-console-cant-locate-backports-shutil-get-terminal-size-and-wont-load
RUN $PIP2 install -I backports.shutil_get_terminal_size

# PyMC
RUN apt-get update && apt-get install -y \
    libfftw3-3
RUN conda install -n python2 -c https://conda.binstar.org/pymc pymc

USER jovyan
COPY custom.js /home/jovyan/.jupyter/custom.js

WORKDIR /home/jovyan/work
