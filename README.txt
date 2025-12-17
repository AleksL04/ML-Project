Sarcasm Detection Model:

Run either predict.py or predict_sarcasm.py using the code:
    python predict.py --input test_data.csv --output predictions.csv

In order to get sarcasm predictions.

training_code.ipynb :
The training code was created and ran through a google colab kernal, in order to take advantage of google colab's GPU.
This was used to speed up training time, however the regular predict.py or predict_sarcasm.py will work strictly on a local machine.
Internet will be required for training and for pip installing dependencies, but running the actual model does not require the internet. 

Warnings:
Use Python 3.12.
pip install the requirements first.