import os

# The path where your 0-100 folders are located
base_path = '.'

# Iterate through all folders (from 0 to 100) and rename both folders and files
for i in range(101):
    # Folder name with zero-padding (e.g., 000, 001, ..., 100)
    folder_name = str(i).zfill(3)
    folder_path = os.path.join(base_path, str(i))  # Current folder (e.g., 0, 1, 10, ..., 100)
    new_folder_path = os.path.join(base_path, folder_name)  # New folder name with zero-padding
    
    # Rename the folder if the names are different
    if folder_path != new_folder_path:
        os.rename(folder_path, new_folder_path)
    
    # Now rename the files inside the folder (both .txt and .png files)
    if os.path.isdir(new_folder_path):
        for filename in os.listdir(new_folder_path):
            if filename.endswith('.txt') or filename.endswith('.png'):
                old_file_path = os.path.join(new_folder_path, filename)
                new_filename = f"{folder_name}{os.path.splitext(filename)[1]}"  # Use zero-padded folder name and preserve extension
                new_file_path = os.path.join(new_folder_path, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)

print("Folders and files renamed successfully!")
