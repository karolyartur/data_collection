#!/bin/bash

curr_user=$(whoami)
#[ "$UID" -eq 0 ] || exec sudo bash "$0" "$@"0

# Install python and pip
sudo apt-get install -y python3.6
sudo apt-get install -y python3-pip
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev

# Install venv
sudo apt-get install -y python3-venv

# Create virtualenv and install pip packages
mkdir -p ../environments
cd ../environments
python3 -m venv data_collection
#sudo chown -R $curr_user obstacle_detection
source data_collection/bin/activate
cd ../data_collection
pip install wheel
pip install -r requirements.txt

cd videoseg/lib/pyflow
python setup.py build_ext -i
echo "Installation complete"