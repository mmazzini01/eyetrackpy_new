import pathlib
import sys

sys.path.append("../..")
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
import re
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
import os

import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import pandas as pd


class EyeTrackingDataUserSet():
    def __init__(
        self, x_screen=1280, y_screen=1024
    ):
        self.x_screen = x_screen
        self.y_screen = y_screen

    def _read_image_file(self, file):
        """
        Read image trial and return dataframe with coordinates of words
        """
        if not isinstance(file, str):
            file = str(file)
        coordinates_trial = {}
        img = cv2.imread(file)
        img = cv2.resize(img, (self.x_screen, self.y_screen))
        d = pytesseract.image_to_data(
            img, output_type=Output.DICT, config="--psm 6"
        )
        n_boxes = len(d["level"])
        counter = 0
        for i in range(n_boxes):
            (x, y, w, h) = (
                d["left"][i],
                d["top"][i],
                d["width"][i],
                d["height"][i],
            )
            if d["text"][i] != "":
                # first version with this join with a ',' of the tokenize box
                # I removed it becouse a lot of times I end with things like "outside,?" instead of original "outside?"
                # text = ','.join([str(x) for x in self.tokenize_sentence(d['text'][i])])
                text = d["text"][i]
                # print(text)
                coordinates_trial[counter] = [counter, text, x, y, x + w, y + h]
                counter += 1
        coordinates_trial = pd.DataFrame.from_dict(
            coordinates_trial,
            orient="index",
            columns=["number", "text", "x1", "y1", "x2", "y2"],
        )

        return coordinates_trial

    def save_coordinates(self, trials, coordinates):
        """
        Save coordinates of words of different trials in csv files"""
        if not isinstance(trials, list):
            trials = [trials]
        for trial in trials:
            self.save_coordinates_trial(trial, coordinates[trial])

    @staticmethod
    def save_coordinates_trial(coordinates_data, file_name, folder):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with data_trial
        coordinates_data.to_csv(
            folder
            + "/"
            + file_name
            + ".csv",
            sep=";",
        )
        return True
    
    @staticmethod
    def search_images_files(folder):
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        file_paths = list(folder.rglob("*"))
        files = {}
        pattern = r"_resp_(\d+\.\d+)"
        for file in file_paths:
            # The regex pattern
            # Perform the match
            match = re.search(pattern, str(file))
            # Check if there is a match and extract the number
            if match:
                trial = match.group(1)
                files[trial] = file

        return files
    
    @staticmethod
    def search_word_coor_files(folder):
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        file_paths = list(folder.rglob("*"))
        files = {}
        pattern = r"word_cor_image_(\d+\.\d+)"
        for file in file_paths:
            # The regex pattern
            # Perform the match
            match = re.search(pattern, str(file))
            # Check if there is a match and extract the number
            if match:
                trial = match.group(1)
                files[trial] = file

        return files
    
    @staticmethod
    def search_word_coor_fixations_files(folder):
        if isinstance(folder, str):
            folder = pathlib.Path(folder)
        file_paths = list(folder.rglob("*"))
        files = {}
        pattern = r"word_cor_image_fixations_(\d+\.\d+)"
        for file in file_paths:
            # The regex pattern
            # Perform the match
            match = re.search(pattern, str(file))
            # Check if there is a match and extract the number
            if match:
                trial = match.group(1)
                files[trial] = file

        return files
    
    @staticmethod
    def _read_coor_trial(file):
        data_trial = pd.read_csv(
            file,
            sep=";",
            index_col=0,
        )
        data_trial = data_trial.loc[
            :, ~data_trial.columns.str.contains("^Unnamed")
        ]
        return data_trial
    
    @staticmethod
    def load_words_fixations_trial(folder_name, trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        try:
            words_fix_trial = pd.read_csv(
                folder_name
                + "/"
                + "vertices"
                + "/word_cor_image_fixations_"
                + str(trial)
                + ".csv",
                sep=";",
            )
            words_fix_trial = words_fix_trial.loc[
            :, ~words_fix_trial.columns.str.contains("^Unnamed")
        ]
            return words_fix_trial
        except:
            return None
        
    @staticmethod
    def save_words_fixations_trial(folder_name, trial, words_fix_trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        
        words_fix_trial.to_csv(
            folder_name
            + "/word_cor_image_fixations_"
            + str(trial)
            + ".csv",
            sep=";",
        )
        return True
       
