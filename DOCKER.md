# Emotion Recognition — Docker Setup

## Build and run (Linux with webcam)

`ash
xhost +local:docker
docker-compose up --build
`

## Train model

`ash
docker run --rm -v \C:\Users\927ni\my\emotion-recognition/data:/app/data emotion-recognition python model/train.py --data_dir /app/data/fer2013
`
