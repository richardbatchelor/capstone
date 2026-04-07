import os
import numpy as np

def collatePreviousSubmissions(functionNo):
    functionIndex = functionNo -1
    assert(functionIndex >= 0)
    assert(functionIndex <= 7)

    #Search for subfolders and find last submission week and results
    week_folders = [d for d in os.listdir("submission_results") if d.startswith("week") ]
    week_folders.sort(key=lambda w: int(w.replace("week", "")))
    last_week = week_folders[-1]
    week_dir = os.path.join("submission_results", last_week)


    def load_file(file_name):
        file_path = os.path.join(week_dir, file_name)
        all_lines = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse the trusted Python literals
                parsedData = eval(line, {"array": np.array, "np": np})
                all_lines.append(parsedData)
        return all_lines

    allSubmissions = load_file("inputs.txt")
    allResults = load_file("outputs.txt")

    #Extract the functionNo inputs and outputs only and return those arrays only
    functionSubmissions = np.array([line[functionIndex] for line in allSubmissions])
    functionResults = np.array([line[functionIndex] for line in allResults])

    return functionSubmissions, functionResults


def sortByLargestOutputDescending(arr):
    return  arr[np.argsort(arr[:, -1])[::-1]]


def combineInitialDataAndSubmissionsToDate(functionNo):
    assert(functionNo >= 1)
    assert(functionNo <= 8)

    initialInputs = np.load(f'initial_data/function_{functionNo}/initial_inputs.npy')
    initialOutputs = np.load(f'initial_data/function_{functionNo}/initial_outputs.npy')

    submissions, results = collatePreviousSubmissions(functionNo)

    combined_Inputs = np.vstack((initialInputs, submissions))
    combined_Outputs = np.concatenate((initialOutputs, results))

    assert combined_Inputs.shape[1] == initialInputs.shape[1]
    assert combined_Outputs.shape[0] == combined_Inputs.shape[0]

    return combined_Inputs, combined_Outputs

