FROM alpine:latest

LABEL MAINTAINER="Andrew Kyres <andrewkyres117@gmail.com>"

WORKDIR /var/www/

# dumb-init: minimal init system for Linux containers
# musl: standard C library
# lib6-compat: compatibility libraries fro glibc
# linux-headers: commonly needed, and an unusual package name from Alpine.
# build-base: used so we include the basic development packages (gcc)
# bash: so we can access /bin/bash
# git: to ease repo cloning
# ca-certificates: for SSL verification during Pip and easy_install
# freetype: library used to render text onto bitmaps and provides support font-related operations
# libgfortran: contains a Fortran shared library, needed to run Fortran
# libgcc: contains shared code that would be inefficient to duplicate every time, as well as auxiliary helper routines and runtime support
# libstdc++: The GNU standard C++ Library. This package contains an additional runtime Library for C++ programs built with the GNU compiler
# openblas: open source implementation of the BLAS (Basic Linear Algebra Subprograms) API with many hand-crafted optimizations for specific processor types
# libssl1.0: SSL shared libraries
ENV PACKAGES="\
    dumb-init \
    musl \
    libc6-compat \
    linux-headers \
    build-base \
    bash \
    git \
    ca-certificates \
    freetype \
    libgfortran \
    libgcc \
    libstdc++ \
    openblas \
    libssl1.0 \
    "

# numpy: support for large, multi-dimensional arrays and matrices
# matplotlib: plotting library for Python and its numerical mathematics extension NumPy.
# scipy: library used for scientific computing and technical computing
# scikit-learn: machine learning library integrates with Numpy and SciPy
# pandas: library providing high-performance, easy-to-use data structures and ata analysis tools
# nltk: suite of libraries and programs for symbolic and sstatistical natural language processing for English
ENV PYTHON_PACKAGES="\
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    nltk \
    "

# Linking of locale.h as xlocale.h is done to ensure successful install of python numpy package
RUN apk add --no-cache --virtual build-dependencies python3 \
    && apk add --virtual build-runtime \
    build-base python3-dev openblas-dev freetype-dev pkgconfig gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && python3 -m ensurepip \
    && rm -r /usr/lib/python*/ensurepip \
    && pip3 install --upgrade pip setuptools \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && rm -r /root/.cache \
    && pip install --no-cache-dir $PYTHON_PACKAGES \
    && apk del build-runtime \
    && apk add --no-cache --virtual build-dependencies $PACKAGES \
    && rm -rf /var/cache/apk/*

CMD ["python3"]
