#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ML
cd /home/bigmountain87/mim_novel

# Wait for C to finish
while ps aux | grep 'generate_500_C.py' | grep -v grep > /dev/null; do
    sleep 30
done

echo '=== C finished, starting PBTL ===' >> /home/bigmountain87/mim_novel/log_pbtl.txt
python -u step0_screen/pbtl_experiment.py >> /home/bigmountain87/mim_novel/log_pbtl.txt 2>&1

echo '=== PBTL finished, starting B data efficiency ===' >> /home/bigmountain87/mim_novel/log_dataeff_B.txt
python -u step0_screen/data_efficiency_BC.py --struct B >> /home/bigmountain87/mim_novel/log_dataeff_B.txt 2>&1

echo '=== B done, starting C data efficiency ===' >> /home/bigmountain87/mim_novel/log_dataeff_C.txt
python -u step0_screen/data_efficiency_BC.py --struct C >> /home/bigmountain87/mim_novel/log_dataeff_C.txt 2>&1
