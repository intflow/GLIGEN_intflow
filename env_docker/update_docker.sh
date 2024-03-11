#!/bin/bash

#sudo docker login docker.io -u kmjeon -p # Type yourself

sudo docker commit intflow_gligen_H100_0.1 gligen:latest-dev-cuda12.2-cudnn9.0-trt8.6.3-gcc11.4
#sudo docker tag gligen_dev:latest intflow/gligen_dev:latest
#sudo docker push intflow/gligen_dev:latest
