FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get -y update
#RUN apt-get -y install apt-utils
RUN apt-get -y install python3-tk

# Define environment variable
ENV DISPLAY unix:0

# Run app.py when the container launches
CMD ["python3", "/app/mnist_Classifier.py"]
