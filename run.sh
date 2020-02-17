docker run -it --gpus all \
-v $PWD/:/app/  \
-p 9000:8080 \
--rm marciopuga/stylegan2-runway:1.0.0 bash
