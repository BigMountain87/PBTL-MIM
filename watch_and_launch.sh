#\!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ML
cd /home/bigmountain87/mim_novel
LOG=/home/bigmountain87/mim_novel/log_parallel.txt
echo "Watcher started at $(date)" >> $LOG
while ps aux | grep pbtl_B.py | grep -v grep > /dev/null; do sleep 20; done
echo "PBTL_B done at $(date)" >> $LOG
python -u step0_screen/pbtl_C.py >> /home/bigmountain87/mim_novel/log_pbtl_C.txt 2>&1 &
PID_C=$\!
python -u step0_screen/random_baseline.py >> /home/bigmountain87/mim_novel/log_random.txt 2>&1 &
PID_R=$\!
echo "C PID=$PID_C, random PID=$PID_R" >> $LOG
wait $PID_C $PID_R
echo "ALL DONE at $(date)" >> $LOG
