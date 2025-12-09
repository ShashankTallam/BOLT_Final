
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv3D, MaxPooling3D
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import sys

# Set data directory
DATA_DIR = '../collected_data'
MODEL_SAVE_PATH = '../model/model_weights_new.h5'

class DataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=8, dim=(22, 80, 112), n_channels=3, n_classes=10, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.file_paths[k] for k in indexes]
        y = self.labels[indexes]
        X = self.__data_generation(list_IDs_temp)
        return X, to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_paths_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, file_path in enumerate(file_paths_temp):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    data = json.loads(content)
                    data = np.array(data)
                    if data.shape == (22, 80, 112, 3):
                        X[i,] = data
                    else:
                        # Handle mistmatch or resize if necessary
                        X[i,] = np.zeros(self.dim + (self.new_channels,)) # Placeholder
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                X[i,] = np.zeros(self.dim + (self.n_channels,)) # Placeholder
        return X

def get_file_list(data_dir):
    file_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return [], []

    print(f"Scanning data from {data_dir}...")
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for subdir in subdirs:
        if '_' in subdir:
            label = subdir.rsplit('_', 1)[0]
        else:
            label = subdir
            
        file_path = os.path.join(data_dir, subdir, 'data.txt')
        if os.path.exists(file_path):
            file_paths.append(file_path)
            labels.append(label)
            
    return np.array(file_paths), np.array(labels)

def build_model(num_classes, input_shape=(22, 80, 112, 3)):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, (3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # 1. Get File List (Fast)
    file_paths, labels_raw = get_file_list(DATA_DIR)
    
    if len(file_paths) == 0:
        print("No data found.")
        return

    print(f"Found {len(file_paths)} samples.")
    
    # 2. Encode Labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_raw)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")

    # 3. Split
    train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels_encoded, test_size=0.15, random_state=42)

    # 4. Generators
    train_gen = DataGenerator(train_paths, train_labels, batch_size=8, n_classes=num_classes)
    test_gen = DataGenerator(test_paths, test_labels, batch_size=8, n_classes=num_classes)

    # 5. Build Model
    model = build_model(num_classes)
    
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=1,
        verbose=1,
        # use_multiprocessing=True, # Might be faster, but safer False on Windows first
        # workers=4
    )

    if not os.path.exists('../model'):
        os.makedirs('../model')
    
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
