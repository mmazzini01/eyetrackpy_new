from setuptools import setup, find_packages
from setuptools.command.install import install
import os
# Custom install command to download weights
# class CustomInstallCommand(install):
#     def run(self):
#         # Run the standard install process
#         install.run(self)
        
#         # Define files to download
#         files_to_download = [
#             {
#                 "url": "https://drive.google.com/uc?id=1CTiali54Q7zsT25ciY0y0sIIf2jZVbZG&export=download",
#                 "local_path": "eyetrackpy/data_generator/fixations_predictor_trained_2/model.pth",
#             },
#             {
#                 "url": "https://drive.google.com/uc?id=1_skRPxLzlY3d68bKk1cf656pYFc--URt&export=download",
#                 "local_path": "eyetrackpy/data_generator/fixations_predictor_trained_1/T5-tokenizer-BiLSTM-TRT-12-concat-3",
#             },
#         ]

#         # Process each file
#         current_dir = os.getcwd()
#         for file in files_to_download:
#             url = file["url"]
#             local_path = file["local_path"]
#             full_local_path = os.path.join(current_dir, local_path)
#             directory = os.path.dirname(full_local_path)

#             # Ensure the directory exists
#             if not os.path.exists(directory):
#                 print(f"Creating directory: {directory}")
#                 os.makedirs(directory)

#             # Download and save the file
#             print(f"Downloading from {url}...")
#             response = requests.get(url)
#             if response.status_code == 200:
#                 with open(local_path, "wb") as f:
#                     f.write(response.content)
#                 print(f"File saved to: {local_path}")
#             else:
#                 print(f"Failed to download file from {url}. Status code: {response.status_code}")


setup(
    name="eyetrackpy_new",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your dependencies here
        'numpy',
        'pandas',
        "gdown",
        # Add other dependencies as needed
    ],
    #  cmdclass={
    #     'install': CustomInstallCommand,  # Use the custom install command
    # },
    # other metadata like author, description, license, etc.
    author="Angela Lopez and Matteo Mazzini",
    description="A Python package for eye tracking analysis that extend eyetrackpy library also to prompt-response data.",
)

