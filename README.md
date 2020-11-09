# Data_Collection
Scripts for data collection and preparation

This package is for the data preparation for the [obstacle_detction](https://github.com/karolyartur/obstacle-detection) package.
The prepared data can be used as training / validation and test datasets for the OFSNet (Optical Flow Segmentation Network)
used for the detection of moving obstacles in the `obstacle-detection` repository.

The training of the OFSNet can be done with the [OFSNet_training](https://github.com/karolyartur/OFSNet_training) repository.

## Installation procedure
Clone the repository next to the `obstacle-detection` repository in your workspace:

```bash
git clone https://github.com/karolyartur/data_collection
```

The resulting file structure should look like this:
```bash
│  
├── obstacle-detection
├── data_collection
```

After this, change to the root of the repository and run the install script:
```bash
cd data_collection
./scripts/INSTALL.sh
```

This should create a Python virtual environment in ```../environments/data_collection```. If the `obstacle-detection` repository is already
installed on the system, the virtual environment will be created next to the `obstacle-detection` virtual environment.

After the install, all the necessary python packages are installed inside the virtual environment, so after activating it, the scripts can be used:

```bash
source ../environments/data_collection/bin/activate
```

## Usage
When the install is ready and the virtual environment is activated the scripts in the repository can be used. Since the preparation of the training data is a resource intensive process, **this repository is intended to be used on a remote, powerful machine**. After the preparation of the dataset, the training can also be done on the same remote machine, and the resulted fine-tuned OFSNNet can be uploaded to the system connected to the actual robot.

As the data has to be recorded in the same environmentin which the actual robot will work, the recording of the data is done with the `obstacle-detection` package. The resulted `bag` and `txt` files should be copied to the root of this repository. (For more info on the data recording see the [obstacle_detction](https://github.com/karolyartur/obstacle-detection) repository.)

### Data preparation
Before the dataset preparation, the recorded `txt` files containing the robot states with timestamps have to be corrected. This information is used for the egomotion compensation, but as the messages are saved as plain text, the format of the lines can get corrupted. This causes an error in the datset preparation script, so there is a helper script to fix the issues with the recorded `txt` files.

In order to correct a recorded `txt`file use the `fix_data_files.py` script. This scipt has two arguments which specify the names of the input and output files:

 - `ifilename`: Name of the input file (default is `data`)
 - `ofilename`: Name of the output (corrected) file (default is `data_correct`)

 Example usage:
 ```bash
 python fix_data_files.py -ifilename test_1 -ofilename test_1_correct
 ```

 This will read in the file `test_1.txt` from the root of the repository and save the corrected data file as `test_1_correct.txt` also in the root of the repository.

 After the data files have been corrected, make sure, that the corresponding `bag` files and corrected `txt` files have the same names (different extensions).

 It is important to have the correct camera to robot transformation defined in `cam_robot_transform_config.yaml`. Use the transformation from the `obstacle-detection` package. 
 
 Then call the `create_dataset_bag.py` script to create the dataset from the recorded data.

 Example:
 ```bash
 python create_dataset_bag.py -filename test_1 -correct
 ```

 This will create the resulted dataset from `test_1.bag` and `test_1.txt` in a folder called `data`. The optical flow part of the generated dataset is corrected by the egomotion filter.

 The parameters of the script are the following:

  - `filename`: Name of the `bag` and `txt` files to use (default is `data`)
  - `tdir`: Name of the folder in which the created dataset will be saved (default is `data`)
  - `s`: Size of the images (sxs) in the generated dataset (default is `299`)
  - `correct`: Boolean value, whether to use egomotion filter correction for the optical flow or not (default is `false`)