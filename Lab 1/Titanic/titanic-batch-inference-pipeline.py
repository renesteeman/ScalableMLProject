import os
    
LOCAL=True

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    # print(y_pred)
    passenger = y_pred[y_pred.size-1]
    if passenger == 0:
        passenger_url = "https://lh3.googleusercontent.com/drive-viewer/AFDK6gNF4xxysDnR9rP3VtnQ0gqQABJaF9MHN6fx9ndnGEp3HwuAH8X4Qh-ewYGg-DbgtNM0q3mKfBKVNK2wyvZdPkGvnKpE=w1920-h487"
    elif passenger == 1:
        passenger_url = "https://lh3.googleusercontent.com/fife/AAbDypAIrE-Me7Sqj3LvMGiOSqrypr9JZfdlWp46clUTEYaZC6zzZeF0iT2zi49Hrj8B0-I4DNb7ce1bqCAFDbNMhQnPNhtjXmrgfV-aFuGpUc6otrmuBT8mM6xpeyW5ljhTcLKTg2YUaY6E2d_XCj8WHY3BTXJvbU55KyOplRicpcLpElmXW2s4RW9op-3waE4p6uY0fIRphNMfxjCZDpCzlmni-zq3RsT3YL8CZaV_03CfW84ntB4nirFedH6zfuWhgreTdKmw6lqoHWm9QgRji33fJJ4bG9APfQCB2anF9z7GJJ3_pcHr39e4qtwxeKIta8sFUsXPKZE7SEHPNdfO_yGV34vMDxsfBUgbRLNOzw06HAVQuvUk1Oe4DSd1GyZxNWsSzw1D8DZhvZzHxHlFcc27d_qglMEtNQGVhulZFb2A3ZGR9I2Us7jog2UVnpjyGTLBlMUX_-K1WpUSbsoq9aHGb6rOWN4ZyZ5r2c_yJU4NzFNmQDZInKBsZKEK0S7oAm69VQ8RVZ6O_3vgy-7K-r-3pEPnTuDVFEndSwDJvYOulagpCEy0i-LhdToN0C2Va6wBAAocXHgsif9PcuR7IAmql-gk06iBGt1YtveOBNfAakoTqvD-BGPHLARGosKoS0csT_AJENIV3wIgEYr7E6plpNBvUSjpQGvGF3Bv1WUPQxZkEof4VV1loqX1vDHQYo4IPpbLBhyPnK_Ate09BPkDc4qEmqhgREEddDQbxzYF2yLz_U-wt_IGbJfUlBikFK6QA8emZUG06yc3vEl4y7bhA1Ofy4eEySMqKhCsKdNQfwzydTUUhvG3U5D70dXXhF0rZy6HfwuIJ-wneawSyvKScEyHBH07vE1YYoMKZO8HmAYboDbaHg5Vyy4cEH8Q4ORi8_9WDxPaGbmsXlqmeEtpplm6CKCq8Qsbc7vKelIzqe7rxZfffBbEjKrA8pp2eBD21Z9zJu5MmitKgj3DPWKxMsCLyuKz02ple_NT8IIpliGP7IUJ2W30x5xEmuCB8-winTszhSGOY_49tywV4x8oumgGeT6qzqrde1qlZkeOX_I5SxHgP56o3ApiI_sDSjaWOHyTmY3FQfBAq9_zTCosvS9113S2mXdeSSGYPTW_8303TqoXEYrGiEwwRitA_-0r-BgxfbPCk7muZfqpr5sgnjoDsTmcgxHXFWVr91IASGIz_8Igl5llof951CndkvDrJFuHycKNxjifzBqe-41Ivw-v-IJ8lpB28yiIjw6rtoiTflGOnUK-dnrbanykYus3vhitOZ_r1JCUWK9nU6o4WxcHzJONVyhG3Nt_-vk_EHncI_bWjTfKE8FUDH94QuO3bU6R1T-5sJKmueOGpIG-4jlMx1V4LM6ylel3Dv6k_dh9Ah7WCs6IsMRQ-_7lWOt9dCNntEdcm2hc9jRrrBxgMSB-6IJ5c90MXjj-yiYisGH3K6keppoR9iaddtKWkyo3d0VEobsTPRUyMdQ=w1920-h892"
    print("Survival predicted: " + str(passenger))
    img = Image.open(requests.get(passenger_url, stream=True).raw) 
    img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)
    
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()
    # print(df["variety"])
    label = df.iloc[-1]["survived"]
    if label == 0:
        label_url = "https://lh3.googleusercontent.com/drive-viewer/AFDK6gNF4xxysDnR9rP3VtnQ0gqQABJaF9MHN6fx9ndnGEp3HwuAH8X4Qh-ewYGg-DbgtNM0q3mKfBKVNK2wyvZdPkGvnKpE=w1920-h487"
    if label == 1:
        label_url = "https://lh3.googleusercontent.com/fife/AAbDypAIrE-Me7Sqj3LvMGiOSqrypr9JZfdlWp46clUTEYaZC6zzZeF0iT2zi49Hrj8B0-I4DNb7ce1bqCAFDbNMhQnPNhtjXmrgfV-aFuGpUc6otrmuBT8mM6xpeyW5ljhTcLKTg2YUaY6E2d_XCj8WHY3BTXJvbU55KyOplRicpcLpElmXW2s4RW9op-3waE4p6uY0fIRphNMfxjCZDpCzlmni-zq3RsT3YL8CZaV_03CfW84ntB4nirFedH6zfuWhgreTdKmw6lqoHWm9QgRji33fJJ4bG9APfQCB2anF9z7GJJ3_pcHr39e4qtwxeKIta8sFUsXPKZE7SEHPNdfO_yGV34vMDxsfBUgbRLNOzw06HAVQuvUk1Oe4DSd1GyZxNWsSzw1D8DZhvZzHxHlFcc27d_qglMEtNQGVhulZFb2A3ZGR9I2Us7jog2UVnpjyGTLBlMUX_-K1WpUSbsoq9aHGb6rOWN4ZyZ5r2c_yJU4NzFNmQDZInKBsZKEK0S7oAm69VQ8RVZ6O_3vgy-7K-r-3pEPnTuDVFEndSwDJvYOulagpCEy0i-LhdToN0C2Va6wBAAocXHgsif9PcuR7IAmql-gk06iBGt1YtveOBNfAakoTqvD-BGPHLARGosKoS0csT_AJENIV3wIgEYr7E6plpNBvUSjpQGvGF3Bv1WUPQxZkEof4VV1loqX1vDHQYo4IPpbLBhyPnK_Ate09BPkDc4qEmqhgREEddDQbxzYF2yLz_U-wt_IGbJfUlBikFK6QA8emZUG06yc3vEl4y7bhA1Ofy4eEySMqKhCsKdNQfwzydTUUhvG3U5D70dXXhF0rZy6HfwuIJ-wneawSyvKScEyHBH07vE1YYoMKZO8HmAYboDbaHg5Vyy4cEH8Q4ORi8_9WDxPaGbmsXlqmeEtpplm6CKCq8Qsbc7vKelIzqe7rxZfffBbEjKrA8pp2eBD21Z9zJu5MmitKgj3DPWKxMsCLyuKz02ple_NT8IIpliGP7IUJ2W30x5xEmuCB8-winTszhSGOY_49tywV4x8oumgGeT6qzqrde1qlZkeOX_I5SxHgP56o3ApiI_sDSjaWOHyTmY3FQfBAq9_zTCosvS9113S2mXdeSSGYPTW_8303TqoXEYrGiEwwRitA_-0r-BgxfbPCk7muZfqpr5sgnjoDsTmcgxHXFWVr91IASGIz_8Igl5llof951CndkvDrJFuHycKNxjifzBqe-41Ivw-v-IJ8lpB28yiIjw6rtoiTflGOnUK-dnrbanykYus3vhitOZ_r1JCUWK9nU6o4WxcHzJONVyhG3Nt_-vk_EHncI_bWjTfKE8FUDH94QuO3bU6R1T-5sJKmueOGpIG-4jlMx1V4LM6ylel3Dv6k_dh9Ah7WCs6IsMRQ-_7lWOt9dCNntEdcm2hc9jRrrBxgMSB-6IJ5c90MXjj-yiYisGH3K6keppoR9iaddtKWkyo3d0VEobsTPRUyMdQ=w1920-h892"
    print("passenger actual: " + str(label))
    img = Image.open(requests.get(label_url, stream=True).raw)
    img.save("./actual_passenger.png")
    dataset_api.upload("./actual_passenger.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="passenger_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Passenger Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [passenger],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our passenger_predictions feature group has examples of all 3 passengers
    print("Number of different passenger predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True survived', 'True died'],
                             ['Pred survived', 'Pred died'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different passenger predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different passenger predictions")


if __name__ == "__main__":
    if LOCAL == True :
        g()