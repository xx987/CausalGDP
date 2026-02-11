## CausalGDP: Causality-Guided Diffusion Policies for Reinforcement Learning


Xiaofeng Xiao, Xiao Hu, Gilbert Yang Ye and Xubo Yue <br>
https://arxiv.org/abs/2602.09207


## Reference

Parts of this implementation are inspired by and adapted from the diffusion policy framework proposed in:

- Zhendong Wang, Jonathan J. Hunt, and Mingyuan Zhou.  
  *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*.  
  arXiv preprint arXiv:2208.06193, 2022.

We reuse selected components from the original implementation as modular building blocks, while introducing new causal modeling, guidance mechanisms, and training procedures specific to this work.

Original repository:  
https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL



## Requirements
Please see the ``requirements.txt`` for environment set up details. `Python 3.12` is strongly recommended.


### Running

For running the Gym-Mujoco tasks in our paper, the `.py` file in `Gym_Tasks` can be run directly. 
For example, the Humanoid task can be run using the `Humanoid.py` file:
```.bash
python Gym_Tasks/Humanoid.py
```

For other tasks, you can run them as follows (for example, `maze2d-umaze-v1`)
```.bash
python main.py --env_name maze2d-umaze-v1 --device 0 --ms online
```




