""" extract the features of images with CNN vgg16, modified in the Fully-Connected 2 (4096 features)"""

import os
import numpy as np
import torch
import cv2
import pandas as pd

from tqdm import tqdm
from torch import nn
from sklearn.cluster import KMeans
from torchvision import datasets, models, transforms
from typing import Tuple
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from typing import Tuple

class ExtractFeaturesCnn:
    """Class to extract the features with CNN, more precisally with vgg16"""
    def __init__(self) -> None:
        pass

    def apply_transformation(self) -> dict:
        """Uses the transforms.Compose to defines the data augmentation aproachs that will be used

        Returns:
            dict: Dictionary that is the data transformations in dataset
        """
        self.data_train = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
}

        return self.data_train


    def extract_features_from_cnn(self, path_to_dataset: str, model: models.vgg16, dataloaders):
        """Extract the features from the cnn, with all images. The directory expected had
        classes as subfolder, like: dataset/{sclass_1, class_2, class_3}

        Args:
            path_to_dataset (str): _description_
        """
        
        feature = []
        label = []
        for inputs, labels in tqdm(dataloaders[path_to_dataset]):
            outputs = model(inputs)
            feature.extend(outputs.cpu().detach().numpy())
            label.extend(labels)
        label = np.array(label)

        return [feature,label]

    def load_custom_vgg16(self) -> Tuple[models.vgg16, models.vgg16]:
        """load orginal vgg16 and remove the last two layers, where is the Fully-Connected 2 (4096 features)

        Returns:
            Tuple[models.vgg16, models.vgg16]: Return the modify vgg16 and the original
        """
        custom_model = models.vgg16(pretrained=True)
        custom_model.classifier= nn.Sequential(*list(custom_model.classifier.children())[:-2]) 

        return custom_model, models.vgg16(pretrained=True)

    def prepare_dataloders(self, path_to_dataset: str, data_transforms: dict) -> None:
        """Preparate the dataloaders before to start the extraction of features"""
        image_datasets = {x: datasets.ImageFolder(os.path.join(path_to_dataset, x), data_transforms[x]) for x in ['train', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'test']}
        class_names = image_datasets['train'].classes
        inputs, _ = next(iter(dataloaders['train']))

        return dataloaders, inputs, class_names

class ExtractFeaturesBagOfVisualWords:
    """Extract the features using the baf of visual words"""
    def __init__(self, num_images) -> None:
        self.num_images = num_images

    def load_dataset(self, path_to_dataset: str) -> Tuple(dict, list):
        """Load the dataset as array

        Args:
            path_to_dataset (str): The path to the dataset

        Returns:
            _type_: None
        """
        images_dict = {}
        labels = []
        for label in tqdm(os.listdir(path_to_dataset)):
            images_list = []
            count = 0
            folder_path = f"{path_to_dataset}{label}"
            for name_image in os.listdir(folder_path):
                img_path = f"{folder_path}/{name_image}"
                img = cv2.imread(img_path)
                if img is None:
                    print(img_path)
                    continue
                if img is not None:
                    images_list.append(img)      
                
                count += 1
                if count > self.num_images:
                    break
                    
            images_dict[label] = images_list
            labels.append(label)

        return images_dict, labels

    def extract_with_sift(self, img: np.array) -> Tuple[list, list]:
        """Extract the keycompoint with SIFT descriptor

        Args:
            img (np.array): Image to extrac keypoints as array 

        Returns:
            _type_: Tuple list with the keypoints and descriptors 
        """
        keypoints, descriptor, = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
        return keypoints, descriptor

    def descriptor_of_images(self, images: dict) -> Tuple[dict, list]:
        """Create the descriptors of images using a list of images

        Args:
            images (list): Dict with the images 

        Returns:
            _type_: Dict with description and descriptor list
        """
        descriptors_dict = {}
        descriptor_list = []
        for label, image_list in tqdm(images.items()):
            descriptors = []
            for image in image_list:
                _, des = self.extract_with_sift(image)
                if des is None:
                    continue
                descriptor_list.extend(des)
                descriptors.append(des)
            descriptors_dict[label] = descriptors
        return descriptors_dict, descriptor_list


    def kmeans_num_visual_words(self, num_visual_words: int, descriptor_list: list) -> float:
        """Clusterizy the descriptors. The classical value 'K' of the  kmeans are the same of num_visual_words

        Args:
            num_visual_words (int): Number of visual of words to use
            descriptor_list (list): Descriptor list to get the visual words

        Returns:
            float: _description_
        """
        kmeans = KMeans(num_visual_words)
        kmeans.fit(descriptor_list)
        centers = kmeans.cluster_centers_
        return centers

    def find_index(self, feature, center):
        count = 0
        ind = 0
        for i in range(len(center)):
            if i == 0:
                count = distance.euclidean(feature, center[i])
            else:
                dist = distance.euclidean(feature, center[i])
                if dist < count:
                    ind = i
                    count = dist
        return ind


    def image_class(self, all_bovw, centers):
        """Create a dict of features"""
        dict_feature = {}
        for key, value in all_bovw.items():
            category = []
            for img in value:
                histogram = np.zeros(len(centers))
                for each_feature in img:
                    ind = self.find_index(each_feature, centers)
                    histogram[ind] += 1
                category.append(histogram)
            dict_feature[key] = category
        return dict_feature


    def create_dataframe_to_dataset(self, bag_of_words):
        """Dataset with keypoints extracted 

        Args:
            bag_of_words (_type_): _description_

        Returns:
            _type_: _description_
        """
        line_value = []
        colum_value = []
        for key, label in enumerate(bag_of_words):
            for hist in bag_of_words[label]:
                colum_value.append(key)
                line_value.append(hist)
        df_dataset = pd.DataFrame(line_value)
        df_dataset['class_name'] = colum_value
        df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
        return df_dataset


    def train_test_data(self, df: pd.DataFrames, test_size: float):
        """Divides the entire dataset in train and test_summary_

        Args:
            df (pd.DataFrames): Dataframe of dataset
            test_size (float): Percent of images to use in test

        Returns:
            _type_: The dataset divided in, x_train, y_train, x_test, y_test
        """
        train, test = train_test_split(df, test_size=test_size)
        x_train = train.drop(['class_name'], axis=1).values
        y_train = train['class_name'].values
        x_test = test.drop(['class_name'], axis=1).values
        y_test = test['class_name'].values
        
        return x_train, y_train, x_test, y_test
