
##Generate docker image
# sudo docker build -t gligen_dev .

container_name="intflow_gligen_H100_0.1"
docker_image="gligen:latest-dev-cuda12.2-cudnn9.0-trt8.6.3-gcc11.4"

sudo docker run -it \
--name=${container_name} \
--net=host \
--privileged \
--ipc=host \
--runtime=nvidia \
--gpus all \
-w /works \
-v /home/ubuntu/intflow/works:/works \
-v /data:/data \
-v /backup:/backup \
${docker_image} bash