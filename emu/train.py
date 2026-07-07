# ---------------------- #
# Train and save emulator
# ---------------------- #

import model 

import load_data
import os


def train_emulator(source_bin, lens_bin):    
    X_train, X_validate, Y_train, Y_validate = load_data.process_data(source_bin, lens_bin)
    
    nn_model = model.build_model(n_in=X_train.shape[1], n_out=Y_train.shape[1], n_nodes=128, learning_rate=5e-5)
    
    history = nn_model.fit(X_train, Y_train,
                        validation_data = (X_validate, Y_validate),
                        epochs=50, 
                        batch_size=32
                        )
    # save the model (create emu/models directory if it doesn't exist)
    os.makedirs('emu/models', exist_ok=True)
    nn_model.save('emu/models/source'+str(source_bin)+'_lens'+str(lens_bin)+'.keras')
    return history

def main():
    train_emulator(source_bin=3, lens_bin=0)

if __name__ == "__main__":
    main()



