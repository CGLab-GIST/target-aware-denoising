docker build -t sig24_target .

docker run \
    --rm \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/host \
    -v /usr/share/nvidia:/usr/share/nvidia \
    --network=host -e NVIDIA_DRIVER_CAPABILITIES=all \
    --privileged \
    --runtime=nvidia \
    -v ${PWD}/example_code:/codes \
    --gpus all \
    -it sig24_target /bin/bash;

 