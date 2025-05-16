from eyetrackpy_new.data_processor.models.eye_tracking_data_image import EyeTrackingDataImage

import pytesseract
import pandas as pd 
import os
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

users_list = ['andre', 'angela','matteo','mireia','sebastian']
sessions_list = [1,2]
for user in users_list:
    for session in sessions_list:
        etdi = EyeTrackingDataImage(
        user=user,
        session=session,
        user_set=0,
        x_screen=1536,
        y_screen=864,
        path="./users"
        )
        fixations, words_fix, info = etdi.asign_fixations_process_words_all()
        etdi.save_fixations(words_fix, fixations_all=fixations, info=info)
        print(f"finished for {user} in session {session}")


'''
features = pd.DataFrame(etdi.compute_entropy_all())
etdi.save_features(features)
'''