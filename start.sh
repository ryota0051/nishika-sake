#!/bin/sh
docker run --gpus all --privileged -it --rm -p 8888:8888 -v ${PWD}:/work nishika-sake
