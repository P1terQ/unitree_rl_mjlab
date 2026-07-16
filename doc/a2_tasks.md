# A2 Rough 与 A2 LocoScan 任务说明

本文档汇总仓库内 A2 粗糙地形任务和迁移后的 A2 LocoScan 任务设置、训练命令与常用验证方式。配置入口主要在 `src/tasks/velocity/config/a2/`。

## 任务入口

| 任务 | Task ID | 用途 | 日志目录 |
| --- | --- | --- | --- |
| A2 Rough | `Unitree-A2-Rough` | mjlab 默认 A2 粗糙地形速度跟踪 | `logs/rsl_rl/a2_velocity/` |
| A2 LocoScan | `Unitree-A2-LocoScan` | 从 IsaacGym 基础 `a2_locoscan` 迁移的地形扫描速度跟踪 | `logs/rsl_rl/a2_locoscan/` |

查看已注册 A2 任务：

```bash
/home/ustc/anaconda3/envs/g1_mjlab/bin/python scripts/list_envs.py --keyword A2
```

## 关键设置对比

| 设置 | A2 Rough | A2 LocoScan |
| --- | --- | --- |
| PD 参数 | hip/thigh `Kp=100, Kd=4`; calf `Kp=150, Kd=6` | 全部关节 `Kp=80, Kd=2` |
| Action scale | `0.25` | `0.25` |
| 初始高度 | `z=0.4` | `z=0.45` |
| 初始关节 | thigh `0.9`, calf `-1.8`, 右 hip `0.1`, 左 hip `-0.1` | hip `0.0`; 前腿 thigh `0.8`; 后腿 thigh `1.0`; calf `-1.5` |
| 指令范围 | x `[-1, 2]`, y `[-1, 1]`, yaw `[-1, 1]`, heading `[-pi, pi]` | x `[0, 0.5]`, y `[-0.25, 0.25]`, yaw `[-1, 1]`, heading `[-1, 1]` |
| 指令采样 | `3-8s`, 站立环境比例 `0.05` | `10s`, 站立环境比例 `0.2` |
| 地形课程 | rough terrain 默认混合，最大初始等级 `5` | LocoScan 地形混合，最大初始等级 `2` |
| 观测 | 标准速度任务观测 + `height_scan` | `command` + 5 帧本体历史 + 154 点高度扫描 |
| 观测维度 | 由默认 velocity 配置生成 | actor `367`, critic `375` |
| RL 配置 | 普通 RSL-RL PPO，`max_iterations=10001`, `save_interval=100` | `LocoScanRunner`/`ActorCriticLocoScan`, `max_iterations=80000`, `save_interval=1000`, lr `5e-4` |

LocoScan 的地形扫描网格为 14 x 11，共 154 条 ray；actor 的高度扫描带噪声和延迟，critic 使用无扰动扫描和 estimator target。

## 训练命令

训练 A2 Rough：

```bash
/home/ustc/anaconda3/envs/g1_mjlab/bin/python scripts/train.py Unitree-A2-Rough \
  --env.scene.num-envs=4096
```

训练 A2 LocoScan：

```bash
/home/ustc/anaconda3/envs/g1_mjlab/bin/python scripts/train.py Unitree-A2-LocoScan \
  --env.scene.num-envs=4096
```

无 CUDA 或只做快速 smoke test 时，可降低环境数量和迭代数：

```bash
WARP_CACHE_PATH=/tmp/warp-cache MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/ustc/anaconda3/envs/g1_mjlab/bin/python scripts/train.py Unitree-A2-LocoScan \
  --gpu-ids None \
  --env.scene.num-envs=8 \
  --agent.max-iterations=2 \
  --agent.logger=tensorboard \
  --agent.upload-model=False
```

## 播放与检查

训练 checkpoint 默认保存为 `logs/rsl_rl/<experiment>/<run>/model_<iter>.pt`。播放 Rough：

```bash
/home/ustc/anaconda3/envs/g1_mjlab/bin/python scripts/play.py Unitree-A2-Rough \
  --checkpoint_file logs/rsl_rl/a2_velocity/<run>/model_<iter>.pt
```

播放 LocoScan：

```bash
/home/ustc/anaconda3/envs/g1_mjlab/bin/python scripts/play.py Unitree-A2-LocoScan \
  --checkpoint_file logs/rsl_rl/a2_locoscan/<run>/model_<iter>.pt
```

## 配置文件索引

- 环境与奖励：`src/tasks/velocity/config/a2/env_cfgs.py`
- PPO 与 runner：`src/tasks/velocity/config/a2/rl_cfg.py`
- LocoScan 网络与算法扩展：`src/tasks/velocity/rl/locoscan.py`
- 任务注册：`src/tasks/velocity/config/a2/__init__.py`
