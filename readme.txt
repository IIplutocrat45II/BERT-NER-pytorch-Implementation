Shopee Industrial Ad-hoc project : Named entity recognition
Author: Siddesh S S

Directory: BERT
.
├── data_load.py
├── experiments
│   ├── map.py
│   ├── pretrained_fasttext
│   │   └── cc.en.300.bin
│   ├── script.py
│   ├── shopee_data.json
│   ├── shopee.json
│   ├── tmp_files
│   │   └── above_thresholds.txt
│   └── train.txt
├── extend.py
├── finetune.sh
├── full_data_trained.pt
├── model.py
├── readme.txt
├── run.sh
├── shopee_data
│   ├── cleaner.py
│   ├── JSON_files
│   │   ├── new_data.json
│   │   └── test_data.json
│   ├── LOGS
│   ├── TEST_DATA
│   │   ├── test_chang_version.txt
│   │   ├── test_hand_labelled.txt
│   │   ├── test.txt
│   │   ├── test_unhand_labelled_5k.txt
│   │   └── test_unhand_labelled.txt
│   ├── test_data.json
│   ├── test.txt
│   ├── train.txt
│   └── valid.txt
├── Shopee Report.pdf
├── test.py
├── test.sh
├── train.py
└── Utils
    ├── JSON_read.py
    ├── __pycache__
    │   └── JSON_read.cpython-36.pyc
    └── requirement.txt

Content:
0. Introduction
1. BERT Model
  1.1 Model Architecture
  1.2 DataLoader & Custom data
2. Training the model
  2.1 Feature based approach
  2.2 Finetune
3. Testing the model
4. Experiments
5. Misc.

0. Introduction:
  a. The required dependencies and its version can be installed using the requirement.txt in the Utils folder.
  b. The directories for the train, test and validation file needs to be changed for all the scripts.

1. BERT Model:
  1.1 Model architecture:
      File: ./model.py
      Info: This file contains the model's architecture (nn class).
  1.2 DataLoader & Custom data:
      File: ./data_load.py
      Info: This file contains the code to convert the input to the required format.
            The custom labels can be added or removed to the 'VOCAB'.

2. Training the models:
  File: ./train.py
  Info: This file contains the code for the train and evaluation functions, the required arguments can be edited. Also the device needs to be changed to GPU if it is trained in a distributed training method.
  2.1 Feature based approach:
      Open run.sh:
        1. Edit the directory in which the checkpoints needs to be stored.
        2. Change the required hyperparameters.
  2.2 Finetune:
      Open fintune.sh:
        1. Edit the directory in which the checkpoints needs to be stored.
        2. Change the required hyperparameters.
  To train the model with feature based approach run the run.sh script.
  To finetune the model run the finetune.sh script.

3. Testing the model:
  File: ./test.py
  Info: This file contains the test functions for evaluting the text code, the respective directories needs to be changed.
                       The code contains a reduntant part in it which is made into a function called 'eval()'.
  Open test.sh:
    1. The required hyperparameters can be changed.

  Run the test.sh script for testing.

4. Experiments:
  Info: This folder contains the fastText experiments files.
  shopee.json --> This file is the json file provided by shopee.
  script.py --> fastText implementation file, the respective directories needs to be changed.
  map.py --> Makes the changes to the labels for the words identified by the fastText.

5. Misc:
  The file JSON_read.py in the folder Utils finds the new words from the tested data.
