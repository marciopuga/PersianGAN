docker run -it --gpus all \
-v $PWD/:/workspace/  \
--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
--rm marciopuga/persiangan-runway:1.0.0  bash
