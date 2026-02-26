# 1) Rutas base (ajústalas si cambian)
root="/mnt/c/Users/Usuario2/Documents/JhojanAndres4E/4DGaussians/output/dynerf/coffee_martini"
src="$root/gaussian_pertimestamp"   # aquí están los time_XXXX.ply
dst="$root/sibr_frames"             # salida en formato SIBR por frame
viewer="./viewers/bin/SIBR_gaussianViewer_app"  # ejecutable del viewer (Linux/WSL)

# 2) Verificaciones básicas
if [ ! -d "$src" ]; then
  echo "No existe $src (carpeta de export)."
  exit 1
fi

if [ ! -f "$root/cameras.json" ]; then
  echo "No se encontró $root/cameras.json. Se necesita para SIBR."
  exit 1
fi

# 3) Crear estructura por frame
mkdir -p "$dst"

shopt -s nullglob
count=0
for ply in "$src"/time_*.ply; do
  base="$(basename "$ply" .ply)"     # time_0000
  id="${base#time_}"                 # 0000
  fdir="$dst/frame_${id}"

  mkdir -p "$fdir/point_cloud"
  cp -f "$ply" "$fdir/point_cloud/point_cloud.ply"
  cp -f "$root/cameras.json" "$fdir/cameras.json"
  count=$((count+1))
done
shopt -u nullglob

echo "Frames preparados: $count en $dst"
