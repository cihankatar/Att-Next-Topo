#!/bin/bash

set -euo pipefail

#SBATCH -p akya-cuda      							# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ckatar          							# Kullanici adi
#SBATCH -J training  								# Gonderilen isin ismi
#SBATCH -o dinoakya-%j.out							# Ciktinin yazilacagi dosya adi
#SBATCH -e dinoakya-%j.err							# Hatalarin yazilacagi dosya adi
#SBATCH --gres=gpu:1        							# train.py tek GPU kullaniyor.
#SBATCH -N 1                							# Gorev kac node'da calisacak?
#SBATCH -n 1                							# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10							# akya-cuda: GPU basina 10 CPU.
#SBATCH --time=3-00:00:00							# akya-cuda maksimum sure siniri.
#SBATCH --mail-type=begin        						# send email when job begins
#SBATCH --mail-type=end          						# send email when job ends
#SBATCH --mail-user=cihan.katar@tau.edu.tr

module purge
module load miniconda3
conda activate segmentation
cd /arf/home/ckatar/Desktop/Repo/Att-Next-TopoLoss
git pull --ff-only origin main
python -c "from torch_topological.nn import WassersteinDistance, CubicalComplex; print('torch_topological import OK')"
python -u train.py
