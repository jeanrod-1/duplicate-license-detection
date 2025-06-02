import os
import shutil

def split_labels_and_copy_images(labels_dir, images_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        num_lines = len(lines)
        if num_lines % 6 != 0: #or num_lines == 6:
            continue  # Skip if it's not a multiple of 6 or already 6
        
        image_name = label_file.replace(".txt", ".jpg")  # Assuming images are .jpg
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {label_file}")
            continue
        
        num_splits = num_lines // 6
        for i in range(num_splits):
            new_label_name = label_file.replace(".txt", f"_{i+1}.txt")
            new_label_path = os.path.join(output_dir, "labels", new_label_name)
            
            with open(new_label_path, "w") as f:
                f.writelines(lines[i*6:(i+1)*6])
            
            new_image_name = image_name.replace(".jpg", f"_{i+1}.jpg")
            new_image_path = os.path.join(output_dir, "images", new_image_name)
            shutil.copy(image_path, new_image_path)
            
            # print(f"Created: {new_label_name} and {new_image_name}")
    
    # Final check: Ensure all labels have exactly 6 lines
    for label_file in os.listdir(os.path.join(output_dir, "labels")):
        label_path = os.path.join(output_dir, "labels", label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
        if len(lines) != 6:
            print(f"Error: {label_file} does not have exactly 6 lines!")

labels_dir = "all_datasets/plates_numbers_and_letters/labels"
images_dir = "all_datasets/plates_numbers_and_letters/images"
output_dir = "all_datasets/processed_plates_numbers_and_letters"
split_labels_and_copy_images(labels_dir, images_dir, output_dir)

