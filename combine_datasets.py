import os
import shutil

def merge_directories(src1, src2, dst):
    for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        src1_path = os.path.join(src1, subdir)
        src2_path = os.path.join(src2, subdir)
        dst_path = os.path.join(dst, subdir)
        
        os.makedirs(dst_path, exist_ok=True)
        
        for src in [src1_path, src2_path]:
            if os.path.exists(src):
                for file_name in os.listdir(src):
                    src_file = os.path.join(src, file_name)
                    dst_file = os.path.join(dst_path, file_name)
                    
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    all_datasets = "all_datasets/cars"
    new_dataset = "new_dataset/cars"
    combined_dataset = "combined_dataset/cars"
    
    merge_directories(all_datasets, new_dataset, combined_dataset)
    print("Merge completado correctamente.")
