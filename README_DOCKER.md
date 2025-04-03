See https://opendrift.github.io/install.html


## choose the version you want to make a docker for
```
git clone https://github.com/OpenDrift/opendrift.git # this will get the latest
cd opendrift/ # where Dockerfile lives
docker build -t simonwp/opendrift:latest .
```

## or a specific tag

```
git clone --branch v.gb.0.0 --single-branch --depth=1 https://github.com/simonweppe/opendrift.git &&\
cd opendrift/
docker build -t simonwp/opendrift:v.gb.0.0 .
```

## then push to dockerhub

```
#retag if necessary
docker tag opendrift:latest simonwp/opendrift:v.gb.0.0
```

```
docker login simon_wp
docker push simonwp/opendrift:v.gb.0.0
```

Docker becomes available here : 


docker build -f Dockerfile -t opendrift:v.gb.0.0 . 
