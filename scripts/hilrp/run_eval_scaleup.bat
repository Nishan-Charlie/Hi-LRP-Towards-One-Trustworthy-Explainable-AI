@echo off
echo Starting lineage_comparison.py for 1000 samples...
C:\Users\nisha\miniconda3\envs\mri-diffuser\python.exe scripts\hilrp\lineage_comparison.py > results\scaleup_comparison.log 2>&1
echo Finished lineage_comparison.py.

echo Starting lineage_naive_legs.py for 1000 samples...
C:\Users\nisha\miniconda3\envs\mri-diffuser\python.exe scripts\hilrp\lineage_naive_legs.py > results\scaleup_naive.log 2>&1
echo Finished lineage_naive_legs.py.

echo Scaleup completely finished!
