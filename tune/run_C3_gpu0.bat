@echo off
set CUDA_VISIBLE_DEVICES=0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set WANDB_MODE=offline
cd /d P:\Nicholas\time_multiplexed_NFA\code

echo C=3 start (GPU0, parallel) %DATE% %TIME% >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt

C:\Anaconda\envs\qpi\python.exe train.py --set C=3 M=1 loss_mode=vote >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_C3.log 2>&1

echo C=3 done (GPU0, parallel) %DATE% %TIME% >> P:\Nicholas\time_multiplexed_NFA\tune\sweep_log.txt
