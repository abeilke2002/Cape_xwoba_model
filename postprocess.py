import pandas as pd 
import numpy as np
from tensorflow import keras

def calc_expected_woba(model_path, cape_csv_path):
    model = keras.models.load_model(model_path)
    data = pd.read_csv(cape_csv_path)
    data = data[~((data['PlayResult'] == "Sacrifice") & (data['ExitSpeed'] < 60))]

    x = np.column_stack((data.ExitSpeed.values, data.Angle.values))
    x = np.expand_dims(x, axis=2)  # Ensure the data has the correct shape

    preds = model.predict(x)
    
    # Add predictions to the DataFrame
    for i in range(preds.shape[1]):
        data[f'prob_class_{i}'] = preds[:, i]

    # cape weights
    weights = np.array([0.0197, 0.89, 1.244, 1.59, 1.97])
    data['woba_pred'] = np.dot(preds, weights)
   
    return data
   

if __name__ == '__main__':
    cape_csv_path = 'cluster/data/combined_data.csv'
    model_path = 'cluster/models/total_bases_model.h5'
    
    data_with_preds = calc_expected_woba(model_path, cape_csv_path)
    data_with_preds = data_with_preds.drop_duplicates()
    data_with_preds.to_csv('cluster/data/tb_probs.csv')
    print('Data Saved')