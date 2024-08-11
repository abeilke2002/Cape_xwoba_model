import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    np.random.shuffle(train_df.values)
    
    train_x = np.column_stack((train_df.ExitSpeed.values, train_df.Angle.values))
    test_x = np.column_stack((test_df.ExitSpeed.values, test_df.Angle.values))
    
    return train_df, test_df, train_x, test_x

def create_and_compile_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(pool_size=1),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Changed from_logits to False
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_x, train_y, batch_size=16, epochs=15):
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
    return model

def evaluate_model(model, test_x, test_y):
    print("EVALUATION")
    model.evaluate(test_x, test_y)
    
    # Generate predictions
    predictions = model.predict(test_x)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(test_y, predicted_classes)
    print("Confusion Matrix")
    print(conf_matrix)
    
    return conf_matrix, predicted_classes

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('cluster/plots/confusion_matrix.png')
    plt.close()

def main(train_path, test_path):
    train_df, test_df, train_x, test_x = load_and_preprocess_data(train_path, test_path)
    input_shape = (train_x.shape[1], 1)
    
    model = create_and_compile_model(input_shape)
    model = train_model(model, np.expand_dims(train_x, axis=2), train_df.tb.values)
    
    conf_matrix, predicted_classes = evaluate_model(model, np.expand_dims(test_x, axis=2), test_df.tb.values)
    plot_confusion_matrix(conf_matrix)
    
    print("Classification Report")
    print(classification_report(test_df.tb.values, predicted_classes))

    model.save('cluster/models/total_bases_model.h5')

if __name__ == "__main__":
    train_path = './cluster/data/train_data.csv'
    test_path = './cluster/data/test_data.csv'
    main(train_path, test_path)
