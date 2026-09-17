@echo off
set CUDA_VISIBLE_DEVICES=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set WANDB_MODE=offline
cd /d P:\Nicholas\time_multiplexed_NFA\code

echo Sweep start %DATE% %TIME% > P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt

C:\Anaconda\envs\qpi\python.exe train.py --set C=1 M=1 loss_mode=vote >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_C1.log 2>&1
echo C=1 done %DATE% %TIME% >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt

C:\Anaconda\envs\qpi\python.exe train.py --set C=2 M=1 loss_mode=vote >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_C2.log 2>&1
echo C=2 done %DATE% %TIME% >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt

C:\Anaconda\envs\qpi\python.exe train.py --set C=3 M=1 loss_mode=vote >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_C3.log 2>&1
echo C=3 done %DATE% %TIME% >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt

echo Sweep complete %DATE% %TIME% >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt
