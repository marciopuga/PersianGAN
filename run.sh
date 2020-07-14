docker run -it --gpus all \
-v $PWD/:/app/  \
-p 8000:9000 \
--rm marciopuga/stylegan2-runway:1.0.0 bash
