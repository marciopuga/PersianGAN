docker run -it --gpus all \
-v $PWD/:/app/  \
-p 8090:8080 \
--rm marciopuga/stylegan2:1.0.0 bash
