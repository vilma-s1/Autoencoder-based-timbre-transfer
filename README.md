# Autoencoder-based-timbre-transfer
MSc AI dissertation project. This project involved the training and testing of an autoencoder architecture inspired by the AUTOVC model implemented by Qian et. al. The model is comprised of two encoders and one decoder, and its purpose is to perform timbre transfer on monophonic instrument data.


#Dependencies

• Python 3

• Numpy

• PyTorch >= v0.4.1

• TensorFlow >= v1.3 (only for tensorboard)

• librosa

• tqdm

• torch openL3


#Training the model

To train the model, run the timbre_transfer.ipynb file and ensure all the training data files 
are in the right folders and within the same directory as the Jupyter notebook. Create a folder 
to store the mel-spectrograms that will be generated during the process.

First run the make_spect.py file to generate mel-spectrograms of the data. 

Then create the timbre embeddings by running the file.

Once the data is transformed into embeddings and spectrograms, run data_loader.py.

The timbre_transfer notebook will allow you to train the model by running all the cells in it. 
However, the code could also be run within an IDE like Pycharm if one has access to a local 
GPU by simply running the main.py function.

Within main.py the training parameters and number of iterations can be altered.

Xian et. al. report that the reconstruction loss should converge to 0.0001. However, using this 
model it should approximately converge to a value in the range of [0.0060, 0.0150].


#Evaluating the model

The model can be evaluated using the evaluation.ipynb Jupyter notebook. Simply construct 
the mel-spectrograms of the test data using the make_spect.py function as before. Once the 
spectrograms are generated, the remaining steps are completed in the notebook by running all 
the cells. Ensure to load the pre-trained model checkpoint (new_trained_autovc.ckpt).
