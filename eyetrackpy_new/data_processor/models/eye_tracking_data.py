import pandas as pd

# import OS module
import os

# Get the list of all files and directories
import pathlib

path = pathlib.Path(__file__).parent.resolve()
import spacy

# create class to analyse eye tracking load_dataset


class EyeTrackingData:
    def __init__(self, user, session=0, user_set=0, trial=None, show=False, path=None):
        self.user = user
        self.show = show
        self.session = session
        self.keyword_general = "TEST_eye_tracker"
        if user_set == 0:
            self.user_set = user
        else:
            self.user_set = user_set
        self.only_trial = trial
        if path is not None:
            self.path = path
        else:
            self.path = (
                str(pathlib.Path(__file__).parent.resolve().parent.resolve()) + "data"
            )
        self.folder_name = "participant_" + str(self.user) + "_" + str(self.user_set)
        if int(self.session) == 0:
            self.session = 1
        self.folder_name += "/session_" + str(self.session)
        self.folder_name = str(self.path) + "/" + self.folder_name

    def _read_general_file(self):
        # search in folder name CSV file that has TEST_eye_tracker inside name
        # save actual path ith pathlib

        for file in os.listdir(self.folder_name):
            # check if file is an excel
            if self.keyword_general in file and "FINAL" in file:
                if ".xlsm" in file:
                    # read excel
                    data = pd.read_excel(self.folder_name + "/" + file)
                    # data = pd.read_csv(folder_name + '/' + file, delimiter=';')
                    return data
                if ".csv" in file:
                    # read csv
                    data = pd.read_csv(self.folder_name + "/" + file, delimiter=";;;")
                    # data = pd.read_csv(folder_name + '/' + file, delimiter=';')
                    return data
        return None

    def _read_fixations_file(self):
        # search in folder name CSV file that has TEST_eye_tracker inside name
        # save actual path ith pathlib
        # self.path = pathlib.Path(__file__).parent.resolve().parent.resolve()

        for file in os.listdir(self.folder_name):
            # check if file is an excel
            if "fixations" in file:
                if ".xlsm" in file:
                    # read excel
                    data = pd.read_excel(self.folder_name + "/" + file)
                    # data = pd.read_csv(folder_name + '/' + file, delimiter=';')
                    return data
                if ".csv" in file:
                    # read csv
                    data = pd.read_csv(self.folder_name + "/" + file, delimiter=",")
                    # data = pd.read_csv(folder_name + '/' + file, delimiter=';')
                    return data
        return None

    def tokenize_sentence(self, text):
        # Load the English language model
        nlp = spacy.load("en_core_web_sm")
        # Process the text with spaCy's tokenizer
        doc = nlp(text)
        # Accessing the tokens
        return [token for token in doc]
