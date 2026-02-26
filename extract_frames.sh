#!/bin/bash

# Verificar que ffmpeg esté instalado
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg no está instalado."
    exit 1
fi

# Verificar que se pasó un argumento
if [ -z "$1" ]; then
    echo "Uso: $0 /ruta/a/la/carpeta"
    exit 1
fi

TARGET_DIR="$1"

# Verificar que la carpeta exista
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: La carpeta $TARGET_DIR no existe."
    exit 1
fi

# Ir a la carpeta destino
cd "$TARGET_DIR" || exit 1

# Procesar videos cam*.mp4
for video in cam*.mp4; do
    [ -e "$video" ] || continue

    base_name="${video%.mp4}"

    echo "Procesando $video..."

    mkdir -p "$base_name/images"

    ffmpeg -i "$video" -start_number 0 "$base_name/images/%04d.png"

    echo "Frames guardados en $base_name/images/"
done

echo "Proceso completado."