Bench checkpoint M3, collimated bench (2026-11-11).
bench_M3_bin1_ep2.pth = code/logs/bench_collimated_2026-11-11/M3_8bit_bin1_ep7/model/epoch=002.pth (best validation so
far; the run was still going when this was copied). bench_M3_bin1_config.json = the run's config, train_LOG.txt its log.

Geometry and training: identical to ckpt_M2/README.txt except M = 3 (3 key / decoder pairs, camera averaged over 3 frames).
  val min-max RMSE by epoch: 8.58e-3, 7.56e-4, 4.91e-4 (ep 2, kept), 6.09e-4, 8.90e-4, ...

Sim (100 a, sim_check_M3/report.txt): min-max 4.08e-4, one global scale + offset (the in-situ reward) 2.61e-4.
The 1000-a test.py numbers will be in code/logs/bench_collimated_2026-11-11/results.csv when the run ends (best.pth = this
epoch unless a later epoch beats it).

Bench numbers, sensitivity and expected camera images: ../sim_check_M3/report.txt (and README.md section 0).
Run: python insitu_TDNFA.py --ckpt ./ckpt_M3/bench_M3_bin1_ep2.pth
