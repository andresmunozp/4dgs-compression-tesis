#!/bin/bash
# Ejecuta desde la raíz del repo 4DGaussians (para que ./viewers/... exista)
cd /mnt/c/Users/Usuario2/Documents/JhojanAndres4E/4DGaussians

viewer="./viewers/bin/SIBR_gaussianViewer_app"

# Encuentra la primera carpeta frame_* dentro de sibr_frames
first=$(ls -d "/mnt/c/Users/Usuario2/Documents/JhojanAndres4E/4DGaussians/output/dynerf/coffee_martini/sibr_frames"/frame_* | sort | head -n 1)

if [ -z "$first" ]; then
  echo "No se encontraron carpetas frame_* en sibr_frames"
  exit 1
fi

"$viewer" -m "$first" --rendering-size 1920 1080
