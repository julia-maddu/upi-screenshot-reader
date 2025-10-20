import os

# ======= CONFIGURE THIS =======
folder_path = r"standardized screenshots\phonepe"  # Replace with the full path to your folder
prefix = "phonepe"
extension = ".jpg"
# ==============================

def rename_jpg_files(folder, prefix, ext):
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(ext)]
        files.sort()  # Optional: alphabetical order

        for index, filename in enumerate(files, start=1):
            new_name = f"{prefix} {index}{ext}"
            src = os.path.join(folder, filename)
            dst = os.path.join(folder, new_name)

            # Avoid overwriting an existing file with the new name
            if os.path.exists(dst):
                print(f"Skipped (name conflict): {dst}")
                continue

            os.rename(src, dst)
            print(f"Renamed: {filename} ➝ {new_name}")

        print("\n✅ Renaming completed.")
    except Exception as e:
        print(f"❌ Error: {e}")

rename_jpg_files(folder_path, prefix, extension)
