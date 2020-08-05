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

---
### ADL Project
This project was developed during the internship period in the university and it was presented as computer science bachelor degree project.

##### Documentation
| | Source |
|:-------------------------:|:-------------------------:|
|Thesis|https://github.com/derogab/adl-thesis|
|Slides|https://github.com/derogab/adl-presentation|

##### Source Code
| | Source |
|:------:|:---------:|
|App|https://github.com/derogab/adl-app|
|Server|https://github.com/derogab/adl-server|
|Api|https://github.com/derogab/adl-api|
