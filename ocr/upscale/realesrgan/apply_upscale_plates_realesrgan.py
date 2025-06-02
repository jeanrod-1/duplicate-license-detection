import subprocess


cmd = [
    "python", "inference_realesrgan.py",
    "-n", "RealESRGAN_x4plus",  # Modelo
    "-i", "../../cropped_plates_processed/images",  # Carpeta de entrada con imágenes
    "-o", "upscaled_plates_realesrgan_processed", # Carpeta de salida con imágenes
    "--outscale", "4",  # Factor de escalado

]

subprocess.run(cmd, check=True)


print("Upscale completado.")
