# README

## Quick Start

如果想使用现成的模型运行游戏，可直接跳转第3步——使用训练的模型play game。

### 1. 关闭图像化界面

打开`Env.m`文件，注释与Viewer有关的行

```matlab
function self = Env(file)
    if exist(file,'file')
        self.loadIni(file);
        self.succeed=1;
    else
        self.succeed=0;
        return;
    end

    self.w = self.mapInfo.w;
    self.h = self.mapInfo.h;
    self.render_st=self.sysInfo.render_st;
    self.obv=Observation();
    self.watertank=WaterTank(self.startPos,rand()+5);
    self.watertank.setSatLevel(self.agentInfo.usat);
%   viewer=Viewer(self.w,self.h);
    self.sp=self.agentInfo.sp;
    self.watertank.setParameter(self.agentInfo.sp,self.agentInfo.ip,self.agentInfo.op,
    		self.agentInfo.noise_level);
%   self.addViewer(viewer);
    self.startRecord();
    self.reset();
end
```

### 2. 运行训练脚本

1. 打开`DQN_training/train.m`

```matlab
policy=DQN(false);
```

传入参数为是否使用之前训练过的模型权重文件（若使用，得注意**输入维度需保持一致**）。

训练策略器的一些可调超参数（定义在`DQN_training/DQN.m`中）如下：

- `n_action`：动作空间大小。默认为均匀量化区间，选取的值越大量化越精细，结果越能逼近目标水位。
- `n_state`：状态空间大小。定义为一个二维数组——[当前时刻水位， 当前时刻目标水位]
- `n_hidden_neurual`：隐藏层神经元数量。越大模型的抽象表达能力越强，但也越复杂。
- `n_update_start`：模型开始更新时间点。决定什么时候开始从经验回放池中抽样批次数据进行神经网络训练。
- `lr`：学习率。值越大，模型训练收敛越快；值越小，模型训练收敛越接近理想值。
- `gamma`：状态函数Q值对未来奖励的折扣。值越大，未来奖励reward计入回报的全职越大。
- `epsilon`：初始贪婪策略系数。影响动作的选择，刚开始设置比较大有利于agent选择到所有的动作，之后再衰减。
- `target_update_freq`：目标网络更新频率。设计DQN知识。
- `batch_size`：单次训练批次大小，越大模型收敛越快，但对cpu性能要求越高
- `network_weights_path`：网络权重路径。

2. 运行`DQN_training/train.m`

生成的模型文件保存在`DQN_training/xxx.mat`目录下。

### 3. 使用训练的模型play game

1. 打开`Env.m`文件，取消注释与Viewer有关的行，见第1步注释行位置。

2. 打开`Policy.m`文件，设置读取模型网络权重路径。

   ```matlab
   classdef Policy < handle
       properties
           n_action = 40                                                      % 动作数量, 需要和训练的模型输入维度对齐
           n_state = 2                                                        % 输入状态数量
           n_hidden_neurual = 20                                              % 隐藏层神经元数量
           actual_actions                                                     % 实际离散动作空间
           qnet                                                               % 状态函数Q网络
           network_weights_path = 'DQN_training/network_weights_bc150_naction40_fc20.mat'    % 网络权重路径
       end
   ```

3. 进入主目录`main.m`，运行文件，得到Score分数（提供的`'DQN_training/network_weights_bc128_naction20_fc15.mat'`应该可以达到9853分）

## TO DO

1. 有些时候训练方向错误，导致模型策略选择过大的action，水位总是保持在30以上的高位。

   解决办法：降低batch_size大小。 

## 优化思路

该算法可进一步调优，提供思路如下：

1. 提高n_action动作空间大小

2. 采用非均匀量化动作空间，例如在目标水位附近动作量化值较小，远离目标水位处动作量化值较大。
   映射动作空间代码位于：

   ```matlab
   % /Policy.m
   % 运行游戏main.m时使用的离线策略
   methods
           function self = Policy()
              ... 
              self.actual_actions = linspace(-10, 15, self.n_action);  % 实际动作空间
              ...
           end
   ```

   ```matlab
   % /DQN_training/DQN.m
   % 训练模型时使用的在线策略
   methods
           function self = DQN(enaleUsePreTrainWeight)
              ...
              self.actual_actions = linspace(-10, 15, self.n_action);  % 实际动作空间
              ...
   ```

   3. 增减隐藏层层数
   4. 调整其他超参数，例如优化器、目标网络更新频率……

## 设计思路

### MDP决策过程

- 状态空间

$$
s = [当前时刻水位, 当前时刻目标水位]
$$

- 动作空间

$$
a = [-10, step, 2\times step, \cdots, 15]
$$

- 奖励

$$
reward = \abs{当前水位 -当前目标水位}
$$

### Deep Q Network

- 损失函数（均方误差）

$$
loss = \arg \min_\omega \frac{1}{2N}\sum^N_{i=1}\left [ Q_\omega(s_i, a_i) - \left(r_i + \gamma \max_{a^`}Q_{\omega}\left(s^`_i, a^`_i\right)\right) \right]
$$

其中$s^`_i和a^`_i$代表下一时刻的状态和动作。

### 经验回放

**目的：用于加快模型收敛速度。**

> 在一般的有监督学习中，假设训练数据是独立同分布的，我们每次训练神经网络的时候从训练数据中随机采样一个或若干个数据来进行梯度下降，随着学习的不断进行，每一个训练数据会被使用多次。在原来的 Q-learning 算法中，每一个数据只会用来更新一次值。为了更好地将 Q-learning 和深度神经网络结合，DQN 算法采用了**经验回放**（experience replay）方法，具体做法为维护一个**回放缓冲区**，将每次从环境中采样得到的四元组数据（状态、动作、奖励、下一状态）存储到回放缓冲区中，训练 Q 网络的时候再从回放缓冲区中随机采样若干数据来进行训练。这么做可以起到以下两个作用。
>
> （1）使样本满足独立假设。在 MDP 中交互采样得到的数据本身不满足独立假设，因为这一时刻的状态和上一时刻的状态有关。非独立同分布的数据对训练神经网络有很大的影响，会使神经网络拟合到最近训练的数据上。采用经验回放可以打破样本之间的相关性，让其满足独立假设。
>
> （2）提高样本效率。每一个样本可以被使用多次，十分适合深度神经网络的梯度学习。

### 目标网络

**目的：提供神经网络训练的标签、期望输入值**

> DQN 算法最终更新的目标是让$\Q_\omega(s,a)$逼近$r_i + \gamma \max_{a^`}Q_{\omega}\left(s1, a1 \right)$，由于 TD 误差目标本身就包含神经网络的输出，因此在更新网络参数的同时目标也在不断地改变，这非常容易造成神经网络训练的不稳定性。为了解决这一问题，DQN 便使用了**目标网络**（target network）的思想：既然训练过程中 Q 网络的不断更新会导致目标不断发生改变，不如暂时先将 TD 目标中的 Q 网络固定住。为了实现这一思想，我们需要利用两套 Q 网络。

### DQN算法具体流程

<img src="C:\Users\28692\Desktop\Watertank - 副本\Watertank\DQN_training\pics\1.png" alt="1" style="zoom:50%;" />

### Reference

[DQN 算法 (boyuai.com)](https://hrl.boyuai.com/chapter/2/dqn算法)

