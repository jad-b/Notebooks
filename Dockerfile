FROM ipython/scipyserver

EXPOSE 8888

# Install custom pip packages
RUN pip install -U \
    mpld3 \
    pip

