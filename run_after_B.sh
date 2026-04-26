#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ML
cd /home/bigmountain87/mim_novel

# Wait for B to finish
while ps aux | grep 'generate_500_B.py' | grep -v grep > /dev/null; do
    sleep 30
done

echo '=== B finished, starting C ===' >> /home/bigmountain87/mim_novel/log_gen_C.txt
python -u step0_screen/generate_500_C.py >> /home/bigmountain87/mim_novel/log_gen_C.txt 2>&1
