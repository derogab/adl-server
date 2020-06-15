FROM python:3

# Create app directory
WORKDIR /usr/src/app

# Create useful directories
RUN mkdir /datasets /models

# Install app dependencies
RUN pip install tensorflow keras h5py pandas sklearn scipy numpy

# Copy app 
COPY . .

# Run the app
ENTRYPOINT [ "python", "./server.py" ]
CMD []