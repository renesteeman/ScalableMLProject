import os
    
LOCAL=True

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    iris_df = pd.read_csv("titanic.csv")
    iris_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["Pclass", "Sex", "Age", "Parch"], 
        description="Titanic passenger survival dataset")
    iris_fg.insert(iris_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
