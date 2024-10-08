#!/bin/bash

# This script will initialize a new project

echo 'Creating New Project'
echo ''

echo 'Attempting to load python module'
module load python

echo 'Creating Virtual Environment'
echo ''
python -m venv env

echo 'Activating newly created python environment'
source env/bin/activate

pip install --upgrade pip==21.0

pip install numpy==1.26.4 tensorflow==2.9.0 matplotlib==3.8.0 deepxde==1.10.0

