classdef Policy < handle
    properties
        n_action = 50                                                      % 动作数量, 需要和训练的模型输入维度对齐
        n_state = 2                                                        % 输入状态数量
        n_hidden_neurual = 20                                              % 隐藏层神经元数量
        actual_actions                                                     % 实际离散动作空间
        qnet                                                               % 状态函数Q网络
        network_weights_path = 'DQN_training/network_weights_bc160_nactions50_fc20.mat'    % 网络权重路径
    end

    methods
        function self = Policy()
            % 定义超参数
%            self.n_action = n_action;    % 将动作离散量化为ustat/n_action，即[0, ustat/n_action, 2*ustat/n_action, 3*ustat/n_action, ...]
%            self.n_state = n_state;      
           self.actual_actions = linspace(-10, 15, self.n_action);  % 实际动作空间
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
           tmp_net = load(self.network_weights_path);
           self.qnet.Learnables = tmp_net.net.Learnables;
        end
            
        function [action] = action(self, observation)
            state = dlarray([observation.agent.H, observation.targetHeight]', 'CBT');
            [~, u] = max(predict(self.qnet, state));
            u = extractdata(u);
            action = self.actual_actions(u);       % 从动作区间中返回动作，即开水阀强度值
        end
    end
end