classdef Qnet < matlab.mixin.Copyable
    properties
        net
        layers
        options
    end
    
    methods
        function self = Qnet(n_state, n_action)
            self.layers = [
                sequenceInputLayer(n_state,"Name","sequence")
                fullyConnectedLayer(10,"Name","fc")
                reluLayer("Name","relu")
                fullyConnectedLayer(n_action,"Name","fc_1")
                regressionLayer("Name","regression")];
            self.options = trainingOptions('adam', ...   % 选择优化器
                'MaxEpochs', 1, ...               % 设置最大训练周期数
                'MiniBatchSize', 32, ...            % 设置小批量样本大小
                'Shuffle', 'every-epoch', ...       % 每个周期随机打乱数据
                'Verbose', false);                   % 输出训练过程信息

%             self.net = trainNetwork(randn(n_state,1), randn(n_action,1), self.layers, self.options);
        end
        
    end
end
