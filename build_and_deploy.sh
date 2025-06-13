#!/bin/bash


DOCKER_USERNAME="vladakawallet"
IMAGE_NAME="vintor-api"
TAG="latest"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

docker build -t ${FULL_IMAGE_NAME} . 

if [ $? -eq 0 ]; then
    echo "[+] Docker image built successfully!"

    echo "Please login to Docker Hub:"
    docker login

    echo "Pushing image to Docker Hub..."
    docker push ${FULL_IMAGE_NAME}

    if [ $? -eq 0 ]; then
        echo "[+] Image pushed successfully!"
        echo "Your image is now available at: ${FULL_IMAGE_NAME}"
        echo ""
    else
        echo "[-] Failed to push image"
        exit 1
    fi
else 
    "[-] Failed to build Docker image" 
    exit 1
fi        
