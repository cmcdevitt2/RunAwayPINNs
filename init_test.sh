#!/bin/bash

# This script will initialize a new project

echo 'Creating New Project'
echo ''

mkdir models

module load python

echo 'Creating Virtual Environment'
echo ''
python -m venv env

source env/bin/activate

pip install --upgrade pip

pip install numpy tensorflow torch matplotlib deepxde

echo 'Creating Requirements File'
echo ''

pip freeze > requirements.txt

echo 'Creating Job Script'
echo ''
echo '#!/bin/bash

#SBATCH -p
#SBATCH gpu
#SBATCH --gpus=1
#SBATCH --account=cmcdevitt-roy
#SBATCH --qos=cmcdevitt-roy
#SBATCH --mem-per-gpu=80gb
#SBATCH --time=00:30:00

module load python
module load cuda/11.1.0

# activate python environment
source env/bin/activate

nvidia-smi

python example.py $TIME --lr=0.001 --layers=2,32,32,1 --epochs=2500' > job.slurm

echo 'Creating Launch Script'
echo ''

echo "#!/bin/bash" > launch.sh
echo "TIME=\`date +\%s\`" >> launch.sh
echo "mkdir models/\$TIME" >> launch.sh
echo "sbatch --output=models/\$TIME/log.out --export=TIME=\$TIME job.slurm" >> launch.sh

echo "Initialization Complete"