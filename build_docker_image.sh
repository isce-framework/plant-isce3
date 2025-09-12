#!/bin/bash

IMAGE=plant-isce3
t=0.1.2
echo "IMAGE is $IMAGE:$t"

# fail on any non-zero exit codes
set -ex

# build image
docker build --rm --force-rm --network=host -t ${IMAGE}:$t -f docker/Dockerfile .

 # run tests
# docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/mnt" -w /mnt -it --workdir /home/plant_isce3_user/plant-isce3/tests  --network host "${IMAGE}:$t" pytest

# create image tar
docker save ${IMAGE} > docker/dockerimg_plant-isce3_$t.tar

# remove image
docker image rm ${IMAGE}:$t