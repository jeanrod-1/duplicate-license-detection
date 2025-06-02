import os

def get_max_id_from_file(txt_file):
    """Obtiene el ID más grande (primer número de cada línea) de un archivo .txt."""
    max_id = float('-inf')
    
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                try:
                    current_id = int(parts[0])  # Extrae el primer número
                    max_id = max(max_id, current_id)
                except ValueError:
                    print(f"❌ Error al leer la línea en {txt_file}: {line.strip()}")
    
    return max_id if max_id != float('-inf') else None  # Retorna None si no hay números

def get_max_id_from_folder(folder_path):
    """Recorre todos los archivos .txt en la carpeta y encuentra el ID más grande."""
    max_id = float('-inf')
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_file_path = os.path.join(root, file)
                file_max_id = get_max_id_from_file(txt_file_path)
                
                if file_max_id is not None:
                    max_id = max(max_id, file_max_id)
    
    return max_id if max_id != float('-inf') else None

# 📌 Ejemplo de uso
folder_path = "missing_images_all"  
max_id = get_max_id_from_folder(folder_path)

print(f"🔹 ID más grande en toda la carpeta: {max_id}")
