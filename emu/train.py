# ---------------------- #
# Train and save emulator
# ---------------------- #
import numpy as np
import tensorflow as tf

import model 
import load_data

import pickle
import json
import os
import argparse

# folder_path = '/home/s2561233/Documents/lss/ridges/ridges_toolkit'
# # change working directory to folder_path
# os.chdir(folder_path)

METADATA = "emu/data/metadata.json"


def train_emulator(lens_bin, source_bin, model_folder="emu/models"):
    X_train, X_validate, Y_train, Y_validate, \
        shape_noise_train, shape_noise_validate = load_data.process_data(lens_bin, source_bin)

    Y_train_for_loss = np.concatenate(
        [Y_train, shape_noise_train],
        axis=1,
    )

    Y_validate_for_loss = np.concatenate(
        [Y_validate, shape_noise_validate],
        axis=1,
    )
    
    # tf.keras.backend.clear_session()
    nn_model = model.build_model(
         n_in=X_train.shape[1], 
         n_out=Y_train.shape[1], 
         n_nodes=128, 
         learning_rate=5e-5
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = nn_model.fit(
         X_train,
         Y_train_for_loss,
         validation_data = (X_validate, Y_validate_for_loss),
         epochs=70, 
         batch_size=32,
         callbacks = [callback]
         )

    # save the model (create the model directory if it doesn't exist)
    os.makedirs(model_folder, exist_ok=True)
    suffix = str(lens_bin)+'_source'+str(source_bin)
    nn_model.save(os.path.join(model_folder, 'lens'+suffix+'.keras'))
    with open(os.path.join(model_folder, 'traing_history'+suffix+'.pkl'), 'wb') as file:
            pickle.dump(history.history, file)
    return history


def train_all_bin_pairs(metadata_file, model_folder="emu/models"):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    all_pairs = metadata["bin_pairs"]
    for (l,s) in all_pairs:
        print("TRAINING PAIR:", (l, s))
        train_emulator(l, s, model_folder)



def main():
    parser = argparse.ArgumentParser(description='Train emulator models.')
    parser.add_argument('-mf', '--modelfolder', default='emu/models',
        help='Folder where trained models and histories are saved (default: emu/models)')
    args = parser.parse_args()
    train_all_bin_pairs(METADATA, args.modelfolder)

if __name__ == "__main__":
    main()


