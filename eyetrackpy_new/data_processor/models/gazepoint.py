class Gazepoint:
    def preprocess_fixations_trial(self, fixations_trial):
        # better to use: BPOGX, BPOGY
        fixations_trial = fixations_trial.rename(
            columns={"FPOGX": "x", "FPOGY": "y", "FPOGD": "duration", "FPOGID": "ID"}
        )
        fixations_trial["pupil_r"] = round(
            fixations_trial["RPD"] * fixations_trial["RPS"], 4
        )
        fixations_trial["pupil_l"] = round(
            fixations_trial["LPD"] * fixations_trial["LPS"], 4
        )
        fixations_trial = fixations_trial[["ID", "x", "y", "duration", "pupil_r", "pupil_l"]]
        return fixations_trial
    
    def scale_fixations_trial(self, fixations_trial, x_screen, y_screen):
        """
        Scale trial fixations to screen size
        """
        
        fixations_trial["x"] = fixations_trial["x"] * x_screen
        fixations_trial["y"] = fixations_trial["y"] * y_screen
        return fixations_trial