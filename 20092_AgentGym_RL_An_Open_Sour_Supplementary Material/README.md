## Running Tutorial

### Environment Setup

We recommend using CUDA 12.4, PyTorch 2.4, and Python 3.10. First, install the requirements using the following command:
```sh
echo "Preparing environment for agentgym-rl..."
conda create -n agentgym-rl python==3.10 -y
conda activate agentgym-rl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
# for RL
cd AgentGym-RL
pip3 install -e .
# for agentgym
echo "Preparing environment for agentenv..."
git submodule init
git submodule update
cd AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3
```

### Training

For SFT, DPO and AgentEvol, please refer to the `README.md` of [AgentGym](./AgentGym/README.md).

For RL training:

**1. Environment Setup**

Make sure you have the required environments set up (see [Environment Setup section](#environment-setup) above).

**2. Data Preparation**

We will release our datasets in the future.

**3. Launch the environment server**

Please launch the environment server by referring to the `README.md` of [AgentGym](./AgentGym/README.md).

**4. Training**

You can see the training example scripts for each task in the [examples/train](./examples/train) for AgentGym-RL and the ScalingInter-RL. In addition, you may refer to the training parameters configured in those scripts.

```sh
bash webarena_train.sh
```

Most explanations of the arguments can be found in the docs of [verl](https://verl.readthedocs.io/en/latest/examples/config.html). Other key arguments:
* `data.max_prompt_length`: Maximum length of the general task description prompt in the first turn.
* `data.max_response_length`: Maximum total token length of the interaction trajectory (excluding the task prompt).
* `actor_rollout_ref.agentgym.task_name`: Training task name of AgentGym.
* `actor_rollout_ref.agentgym.env_addr`: URL of the AgentGym environment server.
* `actor_rollout_ref.rollout.max_tokens`: Maximum token length of a single response per turn.
* `actor_rollout_ref.rollout.rollout_log_dir`: Directory for storing rollout trajectories.
* `algorithm.rounds_ctrl.type`: Strategy for controlling the maximum number of interaction turns. Options:
  - `fixed`: fixed number of turns.
  - `scaling_inter_stepwise`: number of turns increases at fixed step intervals.
* `algorithm.rounds_ctrl.rounds`: Maximum number of allowed interaction turns.
* `algorithm.rounds_ctrl.steps_scaling_inter`: Frequency (in training steps) to increase the maximum number of turns when using `scaling_inter_stepwise`.

See [AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml](./AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml) for more details.

To launch the AgentGym-RL training, set:

```sh
algorithm.rounds_ctrl.type=fixed \
algorithm.rounds_ctrl.rounds=15 \
```

You can see [examples/train/AgentGym-RL/webarena_train.sh](./examples/train/AgentGym-RL/webarena_train.sh) as an example.

To launch the ScalingInter-RL training, set:

```sh
algorithm.rounds_ctrl.type=scaling_inter_stepwise\
algorithm.rounds_ctrl.steps_scaling_inter=100 \
algorithm.rounds_ctrl.rounds=[10,20,30] \
```

You can see [examples/train/ScalingInter-RL/webarena_train.sh](./examples/train/ScalingInter-RL/webarena_train.sh) as an example.

### Evaluation

**1. Environment Setup**

Make sure you have the required environments set up (see [Environment Setup section](#environment-setup) above).

**2. Data Preparation**

We will release our datasets in the future.

**3. Launch the environment server**

Please launch the environment server by referring to the `README.md` of [AgentGym](./AgentGym/README.md).

**4. Evaluation**

You can see the evaluation example scripts for each task in the `examples/eval`. In addition, you may refer to the evaluation parameters configured in those scripts.

To run the evaluation, you can see `examples/eval/webarena_eval.sh` as an example.

```sh
bash webarena_eval.sh
```

Most explanations of the arguments can be found in the docs of [verl](https://verl.readthedocs.io/en/latest/examples/config.html). See `AgentGym-RL/verl/agent_trainer/config/generation.yaml` for more details.
