import splitfolders
import os


class PreprocessDataset: 
    def __init__(self):
        pass
    
    def rename_images_with_labels(self, path_to_folder: str) -> None:
        """Rename the images with you correspondent label

        Args:
            path_to_folder (str): _description_
        """
        if not os.path.isdir(path_to_folder):
            print("Directort not found !")
            
        for label in os.listdir(path_to_folder):
            count = 0
            folder_path = path_to_folder + label
            for name_image in os.listdir(folder_path):
                img_path = folder_path + '/' + name_image
                extension_img = img_path.split('.')[-1]
                new_name = f'{folder_path}/{label}_{count}.{extension_img}'
                os.rename(img_path, new_name)
                count += 1
        
        print("Dataset Images Was Renamed")
        
    def create_default_dataset(self, path_to_folder: str, path_to_save: str, percent_of_images: list = [0.75, 0.25, 0.0]):
        """Create a dataset with the structure: train, test, validation. """
        train_percent = percent_of_images[0]
        test_percet = percent_of_images[1]
        val_percent = percent_of_images[2]

        splitfolders.ratio(f"{path_to_folder}",
                       output=f"{path_to_save}",
                       seed=42,
                       ratio=(train_percent, val_percent, test_percet),
                       group_prefix=True,
                       move=False
                       )
        
