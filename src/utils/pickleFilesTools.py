import os
import pickle

def saveProcessedItemToPickle(processed_item, save_directory_path):
    """
    Save a processed item to a pickle file.

    Parameters:
    processed_item (dict): The processed item to save.
    save_directory_path (str): The path to the directory where the pickle file will be saved.

    """

    # Create the directory if it does not exist
    if not os.path.exists(save_directory_path):
        raise ValueError(f"The directory {save_directory_path} does not exist")
    else:
        # Create the file path
        file_path = save_directory_path + "/sample"+str(processed_item['spokenMNISTindex'])+".pickle"

        # Write the processed_item to the pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(processed_item, file)

        print(f"Processed item saved to {file_path}")


def loadProcessedSample(folder_path):
    """
    Load all pickle files in a folder and return a list of the loaded data.

    Parameters:
    folder_path (str): Path to the folder containing the pickle files.

    Returns:
    list: List of loaded data.
    
    """
    pickle_files = [file for file in os.listdir(folder_path) if file.endswith(".pickle")]

    data_list = []

    for file in pickle_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)

    return data_list