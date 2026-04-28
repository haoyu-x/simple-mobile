## Policy Rolout

Download the checkpoint:
```bash
conda activate robodiff
cd ~/simple-mobile/diffusion_policy/data
gdown "https://drive.google.com/uc?id=1BdChVYSGSJCkxVNx5tCyLtZMYB_SSXfz"
```


Rollout:
```bash
conda activate robodiff
cd ~/simple-mobile/simple_mobile/tidybot2
python rollout.py --sim --ckpt-path ../../diffusion_policy/data/epoch\=0300-train_loss\=0.004.ckpt
```
