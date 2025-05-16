import pandas as pd
import matplotlib.pyplot as plt

# import OS module
import os

# Get the list of all files and directories
import pathlib

path = pathlib.Path(__file__).parent.resolve()
from eye_tracking_data import EyeTrackingData


# create class to analyse eye tracking load_dataset
class EyeTrackingDataText(EyeTrackingData):
    def __init__(self, user, session, trial=None, show=False):
        super().__init__(user, session, trial, show)
        self.colors = ["red", "blue", "green", "yellow"]
        self.data = self._read_general_file()

    def read_answer_file(self, trial):
        for file in os.listdir(self.folder_name + "/" + "vertices"):
            if (
                "_" + str(trial) in file
                and "analysed" not in file
                and "word_cor" not in file
            ):
                self.answer_file = self.folder_name + "/" + "vertices" + "/" + file
                data_answer = pd.read_csv(self.answer_file)
                data_answer["character"] = [i // 4 for i in range(len(data_answer))]
                data_answer["vertex"] = [i % 4 for i in range(len(data_answer))]
                data_answer["colors_c"] = [
                    self.colors[(i // 4) % 4] for i in range(len(data_answer))
                ]
                return data_answer
        return None

    def save_answer_file(self, data_answer, data_trial, error=0):
        if error == 1:
            name = self.answer_file[:-4] + "analysed_ERROR" + ".csv"
        else:
            name = self.answer_file[:-4] + "analysed" + ".csv"
        data_answer.to_csv(name, index=False)
        # create_csv with data_trial
        data_trial.to_csv(
            self.folder_name
            + "/"
            + "vertices"
            + "/word_cor_"
            + str(self.trial)
            + ".csv"
        )
        return True

    def find_words(self, data_answer):
        word = 1
        # add column to data_answer name 'word'
        data_answer["word"] = 0
        vertex_0_x = -100
        for character in data_answer["character"].unique():
            # check if Column X of vertex 0 from character is the same as Column X of vertex 1 from character
            vertex_0_x_pre = vertex_0_x
            vertex_0_x = data_answer[
                (data_answer["character"] == character) & (data_answer["vertex"] == 0)
            ]["Column_X"].values[0]
            vertex_0_y = data_answer[
                (data_answer["character"] == character) & (data_answer["vertex"] == 0)
            ]["Column_Y"].values[0]
            vertex_1_y = data_answer[
                (data_answer["character"] == character) & (data_answer["vertex"] == 1)
            ]["Column_Y"].values[0]

            if vertex_0_x < 0 and vertex_0_x_pre > 0:  # line break
                word += 1

            if abs(vertex_0_y - vertex_1_y) < 5.1:  # blanc space
                data_answer.loc[data_answer["character"] == character, "word"] = 0

                word += 1
            else:
                data_answer.loc[data_answer["character"] == character, "word"] = word
        data_answer["colors_w"] = data_answer["word"] % 4
        # substitute value per position in self.colors
        data_answer["colors_w"] = data_answer["colors_w"].map(
            dict(zip(data_answer["colors_w"].unique(), self.colors))
        )

        return data_answer

    def analyse_vertices_answers(self, trial, show=False):
        trial_data = self.data[self.data["n_resp"] == trial]
        self.trial = trial
        data_answer = self.read_answer_file(trial)
        data_answer = self.find_words(data_answer)
        print("#######################")
        print(trial)
        print(trial_data["resp_text"].values[0])
        answer_text = self.tokenize_sentence(trial_data["resp_text"].values[0])
        data_answer, words_coordinates, error = self.join_words_tokens(
            answer_text, data_answer
        )
        self.save_answer_file(data_answer, words_coordinates, error)

        if show == True:
            plt.scatter(
                data_answer["Column_X"],
                data_answer["Column_Y"],
                color=data_answer["colors_w"],
            )
            plt.show()
            # plt.scatter(data_answer['Column_X'], data_answer['Column_Y'], color=data_answer['colors_c'])
            # plt.show()

        return True

    def join_words_vertices(self):
        trials = self.data["n_resp"].unique()
        # delete nan value in trials
        trials = trials[~pd.isnull(trials)]
        for trial in trials:
            if self.only_trial == None or float(trial) == self.only_trial:
                self.analyse_vertices_answers(trial, self.show)

    def add_words_coordinates_results(
        self, token_id, eye_id, text, words_coordinates, data_answer
    ):
        words_coordinates[token_id] = {}
        words_coordinates[token_id]["text"] = text
        words_coordinates[token_id]["max_x"] = data_answer[
            data_answer["word"] == eye_id
        ]["Column_X"].max()
        words_coordinates[token_id]["min_x"] = data_answer[
            data_answer["word"] == eye_id
        ]["Column_X"].min()
        words_coordinates[token_id]["max_y"] = data_answer[
            data_answer["word"] == eye_id
        ]["Column_Y"].max()
        words_coordinates[token_id]["min_y"] = data_answer[
            data_answer["word"] == eye_id
        ]["Column_Y"].min()
        return words_coordinates

    def order_words(self, data_answer):
        new_number = 1
        for i in data_answer["word"].unique():
            if i > 0:
                data_answer.loc[data_answer["word"] == i, "word"] = new_number
                new_number += 1
        return data_answer

    def join_words_tokens(self, answer, data_answer):
        # iterate over data_answer.word differente values
        i_answer, error, i = 0, 0, 1
        words_coordinates = {}
        data_answer = self.order_words(data_answer)

        while i < len(data_answer["word"].unique()):
            # check indexError

            try:
                answer[i_answer].is_punct
            except IndexError:
                # sentence has ended
                error = 1
                break

            if i != 0:
                print("i token", i_answer, answer[i_answer], "i eye", i)
                try:
                    is_float = float(str(answer[i_answer].text)) % 1 > 0
                except:
                    is_float = 0

                if answer[i_answer].is_punct:
                    # check if the 8 previous rows have been detected as blanck space (one should be the puntuaction sign)
                    try:
                        data_word = data_answer[data_answer["word"] == i]
                        word_index = data_word.index[0]
                        if (
                            sum(
                                data_answer.iloc[word_index - 8 : word_index][
                                    "word"
                                ].values.tolist()
                            )
                            == 0
                        ):
                            i_answer += 1
                        elif len(answer[i_answer]) == 1 and len(
                            data_answer[data_answer["word"] == i]
                        ) / 4 == len(answer[i_answer + 1]):
                            # examples like self-improvement
                            # we want to skip - and is not detected as punct
                            i_answer += 1
                    except IndexError:
                        True

                # count number of rows that have the word i
                n_characters_eye = len(data_answer[data_answer["word"] == i]) / 4
                # count caracters of i_answer text
                n_characters = len(answer[i_answer])
                # ---------------------------------------------------------------------------------
                # the token is a float number and the is separated in tow different eye tracking words
                # ---------------------------------------------------------------------------------
                if is_float:
                    # this eye word plus the next one are a number
                    words_coordinates[i_answer] = {}
                    data_answer.loc[data_answer["word"] == i, "text"] = str(
                        answer[i_answer]
                    )
                    words_coordinates[i_answer]["text"] = str(answer[i_answer])
                    words_coordinates[i_answer]["max_x"] = max(
                        data_answer[data_answer["word"] == i]["Column_X"].max(),
                        data_answer[data_answer["word"] == i + 1]["Column_X"].max(),
                    )
                    words_coordinates[i_answer]["min_x"] = min(
                        data_answer[data_answer["word"] == i]["Column_X"].min(),
                        data_answer[data_answer["word"] == i + 1]["Column_X"].min(),
                    )
                    words_coordinates[i_answer]["max_y"] = max(
                        data_answer[data_answer["word"] == i]["Column_Y"].max(),
                        data_answer[data_answer["word"] == i + 1]["Column_Y"].max(),
                    )
                    words_coordinates[i_answer]["min_y"] = min(
                        data_answer[data_answer["word"] == i]["Column_Y"].min(),
                        data_answer[data_answer["word"] == i + 1]["Column_Y"].min(),
                    )
                    # check if units are inside eye tracking word
                    # example answer[i_answer]  =0.23, answer[i_answer +1] = kg,  data_answer[i] = 0, data_answer[i + 1] = 23kg
                    if (
                        len(str(answer[i_answer].text).split(".")[1])
                        < len(data_answer[data_answer["word"] == i + 1]) / 4
                    ):
                        words_coordinates = self.add_words_coordinates_results(
                            i_answer + 1,
                            i + 1,
                            str(answer[i_answer]).split(".")[1]
                            + " "
                            + str(answer[i_answer + 1]),
                            words_coordinates,
                            data_answer,
                        )
                        data_answer.loc[data_answer["word"] == i + 1, "text"] = (
                            str(answer[i_answer]).split(".")[1]
                            + " "
                            + str(answer[i_answer + 1])
                        )
                        i_answer += 2
                        i += 1
                    else:
                        i_answer += 1
                    i += 1
                    continue

                k, n_characters_sum = 0, n_characters
                # ---------------------------------------------------------------------------------
                # more than one token in one eye tracking word
                # ---------------------------------------------------------------------------------
                while n_characters_eye != n_characters_sum and k < 6:
                    k += 1
                    try:
                        n_characters_sum += len(answer[i_answer + k])
                    except IndexError:
                        # sentence ends, so we asign what we have
                        k = k - 1
                        n_characters_sum = n_characters_eye

                if n_characters_eye == n_characters_sum:
                    # two tokeniced words in one eye tracking detected word
                    # example sample?
                    # example final.
                    text = " ".join(
                        [str(x) for x in answer[i_answer : i_answer + k + 1]]
                    )
                    data_answer.loc[data_answer["word"] == i, "text"] = text
                    words_coordinates = self.add_words_coordinates_results(
                        i_answer, i, text, words_coordinates, data_answer
                    )
                    i_answer += k + 1
                    i += 1
                    continue
                # ---------------------------------------------------------------------------------
                # sometime with mayus in token, eye caracter is plus one
                # ---------------------------------------------------------------------------------
                if (n_characters_eye == n_characters + 1) and str(answer[i_answer])[
                    0
                ].isupper():
                    data_answer.loc[data_answer["word"] == i, "text"] = answer[i_answer]
                    # get the coordinates of the word i
                    words_coordinates = self.add_words_coordinates_results(
                        i_answer,
                        i,
                        str(answer[i_answer]),
                        words_coordinates,
                        data_answer,
                    )
                    i_answer += 1
                    i += 1
                    continue

                # ---------------------------------------------------------------------------------
                # cases like tokens as docker-compose.yml
                # ---------------------------------------------------------------------------------
                k, n_characters_eye_sum = 0, n_characters_eye
                while (
                    n_characters_eye_sum != n_characters
                    and k < 4
                    and (
                        n_characters_eye_sum + 1 != n_characters
                        or str(answer[i_answer])[-1] != "."
                    )
                ):
                    # while n_characters_eye_sum != n_characters and k < 4:
                    k += 1
                    try:
                        # we add one becouse the words in the token should be concatenated by a sign like in token docker-compose.yml
                        n_characters_eye_sum += (
                            len(data_answer[data_answer["word"] == i + k]) / 4 + 1
                        )
                    except IndexError:
                        # sentence ends, so we asign what we have
                        k = k - 1
                        n_characters_eye_sum = n_characters

                if n_characters_eye_sum == n_characters or (
                    n_characters_eye_sum + 1 == n_characters
                    and str(answer[i_answer])[-1] == "."
                ):
                    # two tokeniced words in one eye tracking detected word
                    # example sample?
                    # example e.g.
                    # example final.
                    text = answer[i_answer]
                    max_x = data_answer[data_answer["word"] == i]["Column_X"].max()
                    min_x = data_answer[data_answer["word"] == i]["Column_X"].min()
                    max_y = data_answer[data_answer["word"] == i]["Column_Y"].max()
                    min_y = data_answer[data_answer["word"] == i]["Column_Y"].min()
                    for l in range(0, k + 1):
                        data_answer.loc[data_answer["word"] == i + l, "text"] = text
                        max_x = max(
                            max_x,
                            data_answer[data_answer["word"] == i + l]["Column_X"].max(),
                        )
                        min_x = min(
                            min_x,
                            data_answer[data_answer["word"] == i + l]["Column_X"].min(),
                        )
                        max_y = max(
                            max_y,
                            data_answer[data_answer["word"] == i + l]["Column_Y"].max(),
                        )
                        min_y = min(
                            min_y,
                            data_answer[data_answer["word"] == i + l]["Column_Y"].min(),
                        )

                    words_coordinates[i_answer] = {}
                    words_coordinates[i_answer]["text"] = str(answer[i_answer])
                    words_coordinates[i_answer]["max_x"] = max_x
                    words_coordinates[i_answer]["min_x"] = min_x
                    words_coordinates[i_answer]["max_y"] = max_y
                    words_coordinates[i_answer]["min_y"] = min_y
                    i_answer += 1
                    i += 1 + l
                    continue
                # ---------------------------------------------------------------------------------
                # asignar "a boleo" hasta que alguna coincida. Casos muy complejos con codigo o puntuacion intermedia
                # ---------------------------------------------------------------------------------
                k, j, asign = 0, 0, 0
                n_characters_next, n_characters_eye_next = (
                    n_characters,
                    n_characters_eye,
                )
                while (
                    n_characters_eye_next != n_characters_next
                    and k < 8
                    and j < 8
                    and i + j < len(data_answer["word"].unique()) - 1
                ):
                    # k auments tokens
                    # j auments eye tracking words
                    n_characters_next = len(answer[i_answer + k])
                    n_characters_eye_next = (
                        len(data_answer[data_answer["word"] == i + j]) / 4
                    )
                    print(j, k, n_characters_eye_next, n_characters_next)
                    print(answer[i_answer + k])
                    if (
                        n_characters_eye_next == n_characters_next
                        and n_characters_eye_next > 1
                    ):
                        # asign que next matching word
                        data_answer.loc[data_answer["word"] == i + j, "text"] = answer[
                            i_answer + k
                        ]
                        words_coordinates = self.add_words_coordinates_results(
                            i_answer + k,
                            i + j,
                            str(answer[i_answer + k]),
                            words_coordinates,
                            data_answer,
                        )
                        i_answer += k + 1
                        i += j + 1
                        asign = 1

                        for l in range(min(k, j)):
                            try:
                                data_answer.loc[
                                    data_answer["word"] == i + l, "text"
                                ] = answer[i_answer + l]
                                words_coordinates = self.add_words_coordinates_results(
                                    i_answer + l,
                                    i + l,
                                    str(answer[i_answer + l]),
                                    words_coordinates,
                                    data_answer,
                                )
                            except IndexError:
                                break
                        i_answer += k + 1
                        i += j + 1
                        print("han usado apaño")
                        print(self.trial)
                        print("angela")

                    else:
                        if k > j:
                            j += 1
                        else:
                            k += 1
                # ---------------------------------------------------------------------------------
                # if you haven asign there is an error in this text, and we can join the eye tracking corrdenates with the tokens
                # ---------------------------------------------------------------------------------
                if asign == 0:
                    error = 1
                    print("ERROR")
                    print(self.trial)
                    print(answer)
                    print(answer[i_answer - 1])
                    print(answer[i_answer])
                    print(answer[i_answer + 1])
                    break

        words_coordinates = pd.DataFrame.from_dict(words_coordinates)
        words_coordinates = words_coordinates.transpose()
        return data_answer, words_coordinates, error
