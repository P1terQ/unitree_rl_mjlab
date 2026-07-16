# Repository Guidelines

## Project Structure & Module Organization
This repository combines Python RL training, C++ deployment, and MuJoCo simulation support.
- `src/tasks/` contains registered velocity and tracking task configs, MDP terms, and runners.
- `src/assets/` stores robot XML/MJCF models, meshes, constants, and motion files.
- `scripts/` provides training, play, motion conversion, terrain visualization, and task listing CLIs.
- `deploy/robots/<robot>/` contains C++ controllers, configs, CMake projects, and exported ONNX policies.
- `simulate/` builds the Unitree MuJoCo bridge; `doc/` contains setup guides, media, and license notes.

## Build, Test, and Development Commands
- `pip install -e .` installs the editable Python package and dependencies from `setup.py`.
- `python scripts/list_envs.py` lists registered mjlab task IDs.
- `python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096` starts velocity training and writes logs under `logs/rsl_rl/`.
- `python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/.../model_*.pt` validates a trained policy in MuJoCo.
- `python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1_subject2.csv --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1` converts tracking motion.
- `cd simulate && mkdir -p build && cd build && cmake .. && make -j8` builds the simulator.
- `cd deploy/robots/g1 && mkdir -p build && cd build && cmake .. && make` builds a robot controller.

## Coding Style & Naming Conventions
Python files use two-space indentation, dataclass-based configs, and snake_case names for modules, functions, fields, and CLI flags. Keep task IDs descriptive and robot-prefixed, for example `Unitree-G1-Tracking-No-State-Estimation`. C++ deployment code uses C++17, PascalCase state classes such as `State_RLBase`, and existing brace/namespace style. Avoid formatting churn in vendored `simulate/mujoco/` and `deploy/thirdparty/` code.

## Testing Guidelines
No standalone test suite is currently defined. For Python changes, run the smallest relevant smoke test: `python scripts/list_envs.py`, a low-env training launch, or `scripts/play.py` with a known checkpoint. For C++ or deployment changes, rebuild the affected `deploy/robots/<robot>` target and, when possible, test in `simulate` with `--network=lo` before real hardware.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries, often lowercase, such as `fix tracking deploy` or `add robot As2 and H2`. Keep commits focused by area: task config, asset update, simulator, or deploy target. Pull requests should describe the affected robot/task, list commands run, note generated artifacts or policy files, and include media only for visual behavior changes.

## Security & Configuration Tips
Do not commit build directories, logs, `wandb/`, or generated checkpoints; these are ignored. Treat robot deployment as safety-critical: document network interface choices, policy source, and simulator validation before running controllers with a physical robot.
