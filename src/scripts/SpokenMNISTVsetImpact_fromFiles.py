import multiprocessing as mp

def resevoir(item):
    t = item['t']

    channels = []
    for channel_idx in range(len(item['channels'])):
        V_input = item['channels'][channel_idx]
        # Create an instance of MemDiode
        diode = MemDiode(H0=0,Vset=Vset[wl])
        # Compute the current output
        current_output = []
        H = []
        for V in V_input:
            current_output.append(diode.current_through_GD(V))
            H.append(diode.H)
        # Store the output
        channels.append(np.array(current_output))

    output_item = {
            'spokenMNISTindex': item['spokenMNISTindex'],
            'speaker': item['speaker'],
            'label': item['label'],
            't': t,
            'channels': channels
            }
    
    saveProcessedItemToPickle(output_item, f"/Users/davidgerard/Desktop/Project1/MemReservoir/data/SpokenMNISTVsets/ReservoirOutputs/{int(sample_wavelengths[wl])}")
            
    return output_item





if __name__ == "__main__":
    import sys
    sys.path.append('/Users/davidgerard/Desktop/Project1/MemReservoir/')

    # Import the dataset
    from src.utils.SpokenMNIST.SpokenMnistWrapper import SpokenMnistWrapperAPI
    dataset = SpokenMnistWrapperAPI()

    sample = dataset.get_sample()

    # Process the data
    from src.preprocessing.SpokenMnistInputProcessingTools import processSample

    preprocessor_params={
        "decimation_factor":70,
        "step_factor":0.25,
        "N":100,
        "tau":1.2e-3,
        "voltage_amplitude":5
        }
    voltage_output_params={
        "sample_rate":8000,
        "dt":1e-6,
        "n_cycles":1,
        "normalize":True,
        "plot_coch":False
    }

    from src.utils.pickleFilesTools import loadProcessedSample
    processed_sample = loadProcessedSample("data/SpokenMNISTVsets/ProcessedSamples")

    
    # Define multiple Vsets
    import numpy as np
    N = 7
    sample_wavelengths = np.linspace(500, 1050, N)

    from src.models.nonidealities.Photonic import getVset
    power = 100
    Vset = getVset([power], wavelengths=  sample_wavelengths, Vset0=0.8)

    #Reservoir

    from src.models.memristors.PyMemdiodes import NonVolatileMemDiode as MemDiode
    from src.utils.pickleFilesTools import saveProcessedItemToPickle
    import os
    from src.outputs.RCOutputLayer import FullyConnected, prepareDataForOutputLayer

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import torch

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    import json

    for wl in range(N):
        print("--------------------")
        print(f"Processing sample {wl}")
        if not os.path.exists(f"data/SpokenMNISTVsets/ReservoirOutputs/{int(sample_wavelengths[wl])}"):
            os.mkdir(f"data/SpokenMNISTVsets/ReservoirOutputs/{int(sample_wavelengths[wl])}")

        pool = mp.Pool(mp.cpu_count())
        reservoir_output = pool.map(resevoir, processed_sample)

        pool.close()
        pool.join()
        
        tau = preprocessor_params['tau']
        N = preprocessor_params['N']

        n_channels = len(reservoir_output[0]['channels'])

        n_input = n_channels * N
        n_output = 10

        output_layer = FullyConnected(n_input, n_output,lr = 2e-6,name=f"OutputLayer_{int(sample_wavelengths[wl])}")

        # Prepare the data for the output layer
        X, Y = prepareDataForOutputLayer(reservoir_output, N, tau)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        history = output_layer.train(torch.tensor(X_train).float(), torch.tensor(Y_train).long(), X_test, torch.tensor(Y_test).long(), epochs=500000)
        output_layer.save(f"/Users/davidgerard/Desktop/Project1/MemReservoir/data/SpokenMNISTVsets/Outputlayers")

        # Save the history

        # Convert the history list to JSON format
        history_json = json.dumps(history)

        # Save the JSON data to a file
        with open("/Users/davidgerard/Desktop/Project1/MemReservoir/data/SpokenMNISTVsets/TrainingHistory/"+f"{int(sample_wavelengths[wl])}.json", 'w') as file:
            file.write(history_json)
        
        # Compute the confusion matrix for the training set
        Y_pred_train = np.argmax(output_layer.forward(torch.tensor(X_train).float()).detach().numpy(), axis = 1)

        # Compute the confusion matrix
        cm = confusion_matrix(Y_train, Y_pred_train)
        plt.figure()
        # Plot the confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix on the training set')
        plt.colorbar()

        classes= ['0', '1', '2','3','4','5','6','7','8','9']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add the values in each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig("data/SpokenMNISTVsets/Scores/"+f"{int(sample_wavelengths[wl])}_train.png")
        plt.close()

        # Compute the confusion matrix for the training set


        Y_pred_test = np.argmax(output_layer.forward(torch.tensor(X_test).float()).detach().numpy(), axis = 1)

        # Compute the confusion matrix
        cm = confusion_matrix(Y_test, Y_pred_test)
        plt.figure()
        # Plot the confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix on the test set')
        plt.colorbar()

        classes= ['0', '1', '2','3','4','5','6','7','8','9']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add the values in each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig("data/SpokenMNISTVsets/Scores/"+f"{int(sample_wavelengths[wl])}_test.png")
        plt.close()
