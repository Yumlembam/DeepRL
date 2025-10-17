@echo off
for /L %%s in (1,1,5) do (
  python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 ^
  --exp_name pendulum_default_s%%s ^
  -rtg --use_baseline -na ^
  --batch_size 5000 ^
  --seed %%s
)
