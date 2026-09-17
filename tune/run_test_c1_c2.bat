@echo off
set CUDA_VISIBLE_DEVICES=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d P:\Nicholas\time_multiplexed_NFA\code

echo === Testing C=1 ===
C:\Anaconda\envs\qpi\python.exe test.py --ckpt "P:\Nicholas\time_multiplexed_NFA\code\logs\maj_voting_C_sweep\20260916-1819-M1-C1-K3-Spacings30mm-30mm-5.0mm-30mm-batchsize12-lrslm1e-02-lrlayer1e-02-samples20000-vote\model\best.pth" --csv P:\Nicholas\time_multiplexed_NFA\tune\test_results.csv --sweep maj_voting_C_sweep --label C=1

echo === Testing C=2 ===
C:\Anaconda\envs\qpi\python.exe test.py --ckpt "P:\Nicholas\time_multiplexed_NFA\code\logs\maj_voting_C_sweep\20260917-0049-M1-C2-K3-Spacings30mm-30mm-5.0mm-30mm-batchsize12-lrslm1e-02-lrlayer1e-02-samples20000-vote\model\best.pth" --csv P:\Nicholas\time_multiplexed_NFA\tune\test_results.csv --sweep maj_voting_C_sweep --label C=2

echo === Test sweep done ===
