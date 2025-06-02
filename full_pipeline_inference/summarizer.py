import pandas as pd
from rapidfuzz.distance import Levenshtein
from collections import Counter

# Agrupar placas con distancia de edición ≤ 1
def group_plates_max_distance_one(plate_list):
    print(f"[INFO] Agrupando {len(plate_list)} placas únicas...")
    groups = []
    used = set()

    for i, plate in enumerate(plate_list):
        if i in used:
            continue
        group = [plate]
        used.add(i)
        for j in range(i + 1, len(plate_list)):
            if j in used:
                continue
            if Levenshtein.distance(plate, plate_list[j]) <= 1:
                group.append(plate_list[j])
                used.add(j)
        groups.append(group)
    print(f"[INFO] Se encontraron {len(groups)} grupos de placas similares.")
    return groups

# Eliminar duplicados similares: misma carrocería, color y placa a 1 carácter de distancia
def drop_similar_plates(df):
    df = df.reset_index(drop=True)
    keep_idx = set()
    used_idx = set()

    for i in range(len(df)):
        if i in used_idx:
            continue
        plate_i, type_i, color_i, app_i = df.loc[i, ['plate_text', 'car_type', 'color', 'appearances']]
        group = [i]
        for j in range(i + 1, len(df)):
            if j in used_idx:
                continue
            plate_j, type_j, color_j, app_j = df.loc[j, ['plate_text', 'car_type', 'color', 'appearances']]
            if type_i == type_j and color_i == color_j:
                if Levenshtein.distance(plate_i, plate_j) <= 1:
                    group.append(j)
                    used_idx.add(j)
        best = max(group, key=lambda idx: df.loc[idx, 'appearances'])
        keep_idx.add(best)
        used_idx.update(group)

    return df.loc[sorted(keep_idx)].reset_index(drop=True)

def summarize_similar_plates(csv_path, output_path):
    print(f"[INFO] Leyendo archivo CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filtrar placas vacías
    all_plates = df['plate_text'].fillna('').tolist()
    unique_plates = list(set([p for p in all_plates if p.strip() != '']))
    print(f"[INFO] {len(unique_plates)} placas únicas encontradas (sin contar vacías).")

    plate_groups = group_plates_max_distance_one(unique_plates)

    rows = []
    print("[INFO] Generando resumen por grupo...")
    for idx, group in enumerate(plate_groups):
        sub_df = df[df['plate_text'].isin(group)]

        plate_counter = Counter(sub_df['plate_text'].fillna('').tolist())
        most_common_plate = plate_counter.most_common(1)[0][0] if plate_counter else ''

        car_type_counter = Counter(sub_df['car_type'].dropna().tolist())
        most_common_type = car_type_counter.most_common(1)[0][0] if car_type_counter else ''

        color_counter = Counter(sub_df['color'].dropna().tolist())
        most_common_color = color_counter.most_common(1)[0][0] if color_counter else ''

        image_path = sub_df['image_path'].values[0] if not sub_df['image_path'].empty else ''
        appearances = sub_df['frame'].nunique()

        rows.append({
            'plate_text': most_common_plate,
            'car_type': most_common_type,
            'color': most_common_color,
            'image_path': image_path,
            'appearances': appearances
        })

        print(f"[INFO] Grupo {idx + 1}/{len(plate_groups)} procesado.")

    summary_df = pd.DataFrame(rows)
    print(f"[INFO] Resumen generado con {len(summary_df)} filas.")

    # Eliminar duplicados similares (placa a 1 carácter, mismo color y tipo)
    summary_df = drop_similar_plates(summary_df)

    summary_df.to_csv(output_path, index=False)
    print(f"[INFO] Archivo guardado en: {output_path}")

if __name__ == "__main__":
    print("[INFO] Iniciando procesamiento de placas...")
    summarize_similar_plates(csv_path, output_path)
    
