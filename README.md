# ADL Server
Brain of the ADL project.

### Build 
```bash
docker build -t derogab/adl-server .
```

### Run 
```bash
docker run \
    -dit \
    -p 3000:3000 \
    --restart=always \
    --name adl-server \
    --mount type=bind,source=/path/to/datasets,target=/datasets \
    --mount type=bind,source=/path/to/models,target=/models \
    --mount type=bind,source=/path/to/tmp,target=/tmp \
    derogab/adl-server
```