import os
import sys
import shutil

folder_path = sys.argv[1]

colmap_path = "./colmap_tmp"
images_path = os.path.join(colmap_path, "images")
os.makedirs(images_path, exist_ok=True)
i=0

dir1=os.path.join("data",folder_path)
for folder_name in sorted(os.listdir(dir1)):
    dir2=os.path.join(dir1,folder_name)
    for file_name in os.listdir(dir2):
        if file_name.startswith("frame_00001"):
            i=i+1
            src_path = os.path.join(dir2, file_name)
            dst_path = os.path.join(images_path, f"{i - 1:04d}.png")
            shutil.copyfile(src_path, dst_path) 

print("End！")

# import os
# import sys
# import shutil

# folder_path = sys.argv[1]

# colmap_path = "./colmap_tmp"
# images_path = os.path.join(colmap_path, "images")
# os.makedirs(images_path, exist_ok=True)

# i = 0

# # Usa os.path.join para evitar problemas de rutas en diferentes sistemas operativos
# dir1 = os.path.join("data", folder_path)

# print(dir1)

# # Verificar si el directorio existe antes de procesar
# if not os.path.exists(dir1):
#     print(f"Error: El directorio {dir1} no existe.")
#     sys.exit(1)

# # Recorre las carpetas de cámaras (cam01, cam02, cam03, ...)
# for folder_name in sorted(os.listdir(dir1)):
#     dir2 = os.path.join(dir1, folder_name)

#     print(dir2)

#     if os.path.isdir(dir2):
#         # Asegúrate de que estamos entrando en la carpeta "images" dentro de cada cámara
#         images_subfolder = os.path.join(dir2, "images")

#         # Verificar si la subcarpeta "images" existe dentro de cada cámara
#         if not os.path.exists(images_subfolder):
#             print(f"Advertencia: La subcarpeta 'images' no existe en {dir2}. Omitiendo carpeta.")
#             continue  # Si no existe, omite esta cámara y pasa a la siguiente

#         # Crear la carpeta de salida para cada cámara
#         cam_path = os.path.join(images_path, folder_name)
#         os.makedirs(cam_path, exist_ok=True)

#         cam_images_path = os.path.join(cam_path, "images")
#         os.makedirs(cam_images_path, exist_ok=True)

#         img_counter = 0

#         # Recorre los archivos dentro de la carpeta "images" de cada cámara
#         for file_name in sorted(os.listdir(images_subfolder)):
#             print(f"Procesando archivo: {file_name}")
#             if file_name.startswith("0000"):  # Puedes cambiar esto si necesitas otro criterio
#                 img_counter += 1
#                 src_path = os.path.join(images_subfolder, file_name)

#                 # Renombra las imágenes de la forma "0000.png", "0001.png", etc.
#                 img_name = f"{img_counter - 1:04d}.png"
#                 dst_path = os.path.join(cam_images_path, img_name)

#                 shutil.copyfile(src_path, dst_path)

# print("End!")
