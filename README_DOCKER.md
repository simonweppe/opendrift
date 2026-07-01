See https://opendrift.github.io/install.html


## choose the version you want to make a docker for
```
git clone https://github.com/OpenDrift/opendrift.git # this will get the latest
cd opendrift/ # where Dockerfile lives
docker build -f Dockerfile_from_official -t simonwp/opendrift:latest .
```
## or a specific tag

```
git clone --branch v.gb.0.0 --single-branch --depth=1 https://github.com/simonweppe/opendrift.git &&\
cd opendrift/
docker build -t simonwp/opendrift:v.gb.0.0 .
docker build

git clone --branch v1.14.8.2 --single-branch --depth=1 https://github.com/simonweppe/opendrift.git &&\
cd opendrift/
docker build -f Dockerfile_from_official -t simonwp/opendrift:v1.14.8.2 .


```

## then push to dockerhub

```
#retag if necessary
docker tag opendrift:latest simonwp/opendrift:v.gb.0.0
```

```
#docker login 
docker login simon_wp
docker push simonwp/opendrift:v.gb.0.0
docker push simonwp/opendrift:v.ondemand.0.0
docker push simonwp/opendrift:v1.14.8.2

```

Docker becomes available here : 
https://hub.docker.com/repository/docker/simonwp/opendrift/general


We currently have 

- simonwp/opendrift:v.gb.0.0
- simonwp/opendrift:ondemand
- simonwp/opendrift:v.ondemand.0.0