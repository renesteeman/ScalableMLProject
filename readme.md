The Titanic project consists of a locally executed training step from which the resulting model is uploaded to Hopsworks along with the dataset. This dataset was collected from Kaggle https://www.kaggle.com/competitions/titanic with some changes being made to it, these include dropping the columns for which we did not expect a strong correlation with survivability (name, SibSp (sibling or spouse), ticket number, fare (as this would already be correlated with the class)) and cabin as this had 78% null entries. Furthermore, sex was converted to a numerical value by mapping female to 0 and male to 1, and rows with null values were removed. This data cleaning was performed in Excel as then exported as a csv. 
When a user wants to interact with the model, they can do so by going to https://huggingface.co/spaces/dhruvshettty/titanic-inference and enter the information of a (fictional) passenger to see the model's prediction for their survival. This will send a call to Hopsworks to run the model.
To analyse the performance of the model, one could go to https://huggingface.co/spaces/dhruvshettty/titanic-inference-monitor to see the confusion matrix and recent calls to the model along with two images indicating the previous prediction and the correct result.
The model itself is a simple KNN, which returns a result based on the faith of similar passengers.

titanic-feature-pipeline is responsible for uploading the the processed dataset into Hopsworks.
titanic-feature-pipeline-daily is responsible for uploading a randomly generated passenger and its predicted outcome.
titanic-training-pipeline is responsible for training the model on the dataset and uploading the resulting model onto Hopsworks
titanic-batch-inference-pipeline is responsbile for generating the confusion matrix and uploading the resulting png to Hopsworks 

The HuggingFace spaces contain app.py which will show the interface, the first space shows input fields for the user to enter a (fictional) passenger and the second shows the recent results and the confusion matrix along with 2 images to show the actual and predicted results.
