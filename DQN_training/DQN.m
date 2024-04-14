% https://ww2.mathworks.cn/help/deeplearning/ug/train-network-using-custom-training-loop.html
classdef DQN < handle
    properties
        n_action = 50               % 动作维度（数量）,将动作离散量化为ustat/n_action，即[0, ustat/n_action, 2*ustat/n_action, 3*ustat/n_action, ...]
        n_state = 2                 % 状态维度
        n_hidden_neurual = 20       % 隐藏层神经元数量
        n_update_start = 800        % 模型开始更新时间点
        lr = 0.001                  % 学习率
        gamma = 0.98                % 衰减系数
        epsilon = 1                 % 初始贪婪策略系数，刚开始设置比较大有利于agent选择到所有的动作，之后再衰减
        target_update_freq = 20     % 目标网络更新频率
        batch_size = 160            % 单次训练批次大小，越大模型收敛越快，但对cpu性能要求越高
        network_weights_path = 'DQN_training/network_weights_bc160_naction30_fc20.mat'    % 网络权重路径
    end
    properties (Access = private)
        H                           % 记录当前时刻水位，方便输出
        target_H                    % 记录目标水位，方便输出
        qnet                        % Q网络
        target_qnet                 % 目标Q网络
        n_count = 0                 % 循环计数器
        actual_actions              % 离散动作区间
        u_counts                    % 策略选择的u次数统计器
        % SGDM优化器参数
        initialLearnRate = 0.01
        decay = 0.01
        momentum = 0.9
        velocity = []
    end
    
    methods
        function self = DQN(enaleUsePreTrainWeight)
            % 定义超参数
           self.actual_actions = linspace(-15, 15, self.n_action);
           self.u_counts = zeros(1, self.n_action);
           % 定义Q网路
           layers = [
                sequenceInputLayer(self.n_state,"Name","sequence")
                fullyConnectedLayer(self.n_hidden_neurual,"Name","fc_1")
                reluLayer("Name","relu_1")
                fullyConnectedLayer(self.n_hidden_neurual,"Name","fc_2")
                reluLayer("Name","relu_2")
                fullyConnectedLayer(self.n_action,"Name","fc_3")];
           self.qnet = dlnetwork(layers);
           % 加载之前训练的权重文件
           if enaleUsePreTrainWeight
               tmp_net = load(self.network_weights_path);
               self.qnet.Learnables = tmp_net.net.Learnables;
           end
           self.target_qnet = self.qnet;
        end
        
        function [u, action] = action(self, observation)
            if self.n_count > self.n_update_start
                self.epsilon = self.epsilon / 20;
            end
            self.H = observation.agent.H;
            self.target_H = observation.targetHeight;
            state = dlarray([self.H, self.target_H]', 'CBT');
            if rand() < self.epsilon
                u = randi(self.n_action);   % 获得动作区间索引
            else
%                 net_output = predict(self.qnet, state);
%                 net_forward = forward(self.qnet, state);
                [~, u] = max(predict(self.qnet, state));
                u = extractdata(u);
            end
            action = self.actual_actions(u);       % 从动作区间中返回动作，即开水阀强度值
            self.u_counts(u)  = self.u_counts(u) + 1;   % 记录该动作被选择的次数
            self.n_count = self.n_count + 1;
        end

        function [loss,gradients] = modelLoss(self, qnet, target_qnet, states, actions, next_states, rewards, dones)

            net_output = forward(qnet, states);
            % 计算Q值
%             for i = 1:1:length(actions)
%                 q_values(i) = net_output(actions(i), i);
%             end
            q_values = dlarray(net_output(sub2ind(size(net_output), actions, 1:length(actions))), 'CBT');   % 与上三行等价
            % 计算下个状态的最大Q值
            next_q_values = forward(target_qnet, next_states);
            max_next_q_values = max(next_q_values);
            % 时序差分误差目标
            q_targets = dlarray(rewards + self.gamma .* max_next_q_values .* (1 - dones), 'CBT');

            % Calculate mse loss.
            loss = mse(q_values,q_targets);
            
            % Calculate gradients of loss with respect to learnable parameters.
            gradients = dlgradient(loss,qnet.Learnables);
        
        end

        function update(self, states, actions, rewards, next_states, dones)
            states = dlarray(states, 'CBT');
            actions = dlarray(actions);
            rewards = dlarray(rewards);
            next_states = dlarray(next_states, 'CBT');
            dones = dlarray (dones);
            
            % 使用 dlfeval 跟踪变量并计算损失
            [dqn_loss, gradients] = dlfeval(@self.modelLoss, self.qnet, self.target_qnet, states, actions, next_states, rewards, dones);
            fprintf('LOSS: %f  water lv: %f  target lv: %f\n', dqn_loss, self.H, self.target_H);
            % 更新网络
            [self.qnet,self.velocity] = sgdmupdate(self.qnet,gradients,self.velocity,self.lr,self.momentum);
            if mod(self.n_count, self.target_update_freq) == 0
                self.target_qnet = self.qnet;
            end
        end

        function qnet = get_qnet(self)
            qnet = self.qnet;
        end

    end
end