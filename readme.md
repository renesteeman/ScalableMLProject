The Titanic project consists of a locally executed training step from which the resulting model is uploaded to Hopsworks along with the dataset. This dataset was collected from Kaggle https://www.kaggle.com/competitions/titanic with some changes being made to it, these include dropping the columns for which we did not expect a strong correlation with survivability (name, SibSp (sibling or spouse), ticket number, fare (as this would already be correlated with the class)) and cabin as this had 78% null entries. Furthermore, sex was converted to a numerical value by mapping female to 0 and male to 1, and rows with null values were removed. This data cleaning was performed in Excel as then exported as a csv. 
When a user wants to interact with the model, they can do so by going to https://huggingface.co/spaces/dhruvshettty/titanic-inference and enter the information of a (fictional) passenger to see the model's prediction for their survival. This will send a call to Hopsworks to run the model.
To analyse the performance of the model, one could go to https://huggingface.co/spaces/dhruvshettty/titanic-inference-monitor to see the confusion matrix and recent calls to the model along with two images indicating the previous prediction and the correct result.
The model itself is a simple KNN, which returns a result based on the faith of similar passengers.

titanic-feature-pipeline is responsible for uploading the the processed dataset into Hopsworks.
titanic-feature-pipeline-daily is responsible for uploading a randomly generated passenger and its predicted outcome.
titanic-training-pipeline is responsible for training the model on the dataset and uploading the resulting model onto Hopsworks
titanic-batch-inference-pipeline is responsbile for generating the confusion matrix and uploading the resulting png to Hopsworks 

The HuggingFace spaces contain app.py which will show the interface, the first space shows input fields for the user to enter a (fictional) passenger and the second shows the recent results and the confusion matrix along with 2 images to show the actual and predicted results.

The irish spaces are
https://huggingface.co/spaces/renesteeman/ScalableML_lab1_space1
https://huggingface.co/spaces/renesteeman/ScalableML_lab1_space2

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The original code was split up into a pre-processing, model_training, and inference modules. 

The pre-processing module uses Google Drive for storage, as a lot of storage was required, 200gb storage was purchased. It starts by installing all requirements, then connects to huggingface (used to get the dataset), and loads x% (set to 15 for the 'normal' model to make it fit into memory of a free Colab instance) of the data. It then removes unused columns. To speed up the process, the tiny model was used which is a lot smaller than the small option that the code came with. To pre-process the data, the files are converted into a log-Mel format and downsampled to be in line with Whisper's training data. It then stores the result onto Google Drive. 

The training module connects to the Google Drive to be able to retrieve the pre-processed data. It install all requirements, and then connects to Huggingface where the final model will be stored. It loads the data from Google Drive and loads the model. It uses the wer (word error rate) metric to determine how well the model performs, this equals (substitutions + deletions + inserts)/reference_length. We also turned down the max_steps to speedup the process and lowered the per_device_train_batch_size as we were having issues with a lack of memory, additionally we save the model every 500 steps so we could continue training later in case the notebook would disconnect. Finally, the best version of the model is pushed to Huggingface.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For the final project, we want to build an application that is given a set of articles/tweets/messages and returns you the topic and the stance of each item. This could be used as part of a recommendation that system that would automatically sort a set of articles into different opinions, allowing you to quickly discover the different ways of looking at an issue.

The available data consists of Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representation https://github.com/emilyallaway/zero-shot-stance/tree/master/data, which contains posts along with their topic(s) and stance. 

As this is a large unsolved research problem (both correctly classifying the topic without having a list of possible options, and using stance detection while working with many labels with were not seen in the training data), it might not be possible to build the system exactly as proposed. An alternative would be to use sentiment analysis, which could be done using VADER (https://www.nltk.org/_modules/nltk/sentiment/vader.html) which also wouldn't rely as much on the correct labeling of the topic (as sentiment does not need a target, while stance does)

 
