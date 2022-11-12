import os
    
BACKFILL=False
LOCAL=True

def generate_passenger(survived, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single passenger data as a single row in a DataFrame.
    """
    import pandas as pd
    import random

# PassengerId,Survived,Pclass,Sex,Age,Parch
    df = pd.DataFrame({ 
        "PassengerId": [random.randint(1000, 2000)],
        "Pclass": [random.randint(1, 3)],
        "Sex": [random.choice(['male', 'female'])],
        "Age": [random.randint(0, 100)],
        "Parch": [random.randint(0, 2)]
    })
    df['Survived'] = survived
    return df


def get_random_passenger_data():
    """
    Returns a DataFrame containing one random Titanic passenger data.
    """
    import pandas as pd
    import random

    deceased_df = generate_passenger(0)
    survived_df = generate_passenger(1)

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.randint(0, 1)
    if pick_random == 0:
        titanic_df = deceased_df
        print("Deceased passenger added")
    elif pick_random == 1:
        titanic_df = survived_df
        print("Survived passenger added")

    return titanic_df



def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("data_titanic.csv")
    else:
        titanic_df = get_random_passenger_data()

    iris_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["PassengerId", "Pclass", "Sex", "Age", "Parch"], 
        description="Titanic dataset")
    iris_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
