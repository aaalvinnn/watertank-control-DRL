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
            self.options = trainingOptions('adam', ...   % ѡ���Ż���
                'MaxEpochs', 1, ...               % �������ѵ��������
                'MiniBatchSize', 32, ...            % ����С����������С
                'Shuffle', 'every-epoch', ...       % ÿ�����������������
                'Verbose', false);                   % ���ѵ��������Ϣ

%             self.net = trainNetwork(randn(n_state,1), randn(n_action,1), self.layers, self.options);
        end
        
    end
end
