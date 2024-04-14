% https://ww2.mathworks.cn/help/deeplearning/ug/train-network-using-custom-training-loop.html
classdef DQN < handle
    properties
        n_action = 50               % ����ά�ȣ�������,��������ɢ����Ϊustat/n_action����[0, ustat/n_action, 2*ustat/n_action, 3*ustat/n_action, ...]
        n_state = 2                 % ״̬ά��
        n_hidden_neurual = 20       % ���ز���Ԫ����
        n_update_start = 800        % ģ�Ϳ�ʼ����ʱ���
        lr = 0.001                  % ѧϰ��
        gamma = 0.98                % ˥��ϵ��
        epsilon = 1                 % ��ʼ̰������ϵ�����տ�ʼ���ñȽϴ�������agentѡ�����еĶ�����֮����˥��
        target_update_freq = 20     % Ŀ���������Ƶ��
        batch_size = 160            % ����ѵ�����δ�С��Խ��ģ������Խ�죬����cpu����Ҫ��Խ��
        network_weights_path = 'DQN_training/network_weights_bc160_naction30_fc20.mat'    % ����Ȩ��·��
    end
    properties (Access = private)
        H                           % ��¼��ǰʱ��ˮλ���������
        target_H                    % ��¼Ŀ��ˮλ���������
        qnet                        % Q����
        target_qnet                 % Ŀ��Q����
        n_count = 0                 % ѭ��������
        actual_actions              % ��ɢ��������
        u_counts                    % ����ѡ���u����ͳ����
        % SGDM�Ż�������
        initialLearnRate = 0.01
        decay = 0.01
        momentum = 0.9
        velocity = []
    end
    
    methods
        function self = DQN(enaleUsePreTrainWeight)
            % ���峬����
           self.actual_actions = linspace(-15, 15, self.n_action);
           self.u_counts = zeros(1, self.n_action);
           % ����Q��·
           layers = [
                sequenceInputLayer(self.n_state,"Name","sequence")
                fullyConnectedLayer(self.n_hidden_neurual,"Name","fc_1")
                reluLayer("Name","relu_1")
                fullyConnectedLayer(self.n_hidden_neurual,"Name","fc_2")
                reluLayer("Name","relu_2")
                fullyConnectedLayer(self.n_action,"Name","fc_3")];
           self.qnet = dlnetwork(layers);
           % ����֮ǰѵ����Ȩ���ļ�
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
                u = randi(self.n_action);   % ��ö�����������
            else
%                 net_output = predict(self.qnet, state);
%                 net_forward = forward(self.qnet, state);
                [~, u] = max(predict(self.qnet, state));
                u = extractdata(u);
            end
            action = self.actual_actions(u);       % �Ӷ��������з��ض���������ˮ��ǿ��ֵ
            self.u_counts(u)  = self.u_counts(u) + 1;   % ��¼�ö�����ѡ��Ĵ���
            self.n_count = self.n_count + 1;
        end

        function [loss,gradients] = modelLoss(self, qnet, target_qnet, states, actions, next_states, rewards, dones)

            net_output = forward(qnet, states);
            % ����Qֵ
%             for i = 1:1:length(actions)
%                 q_values(i) = net_output(actions(i), i);
%             end
            q_values = dlarray(net_output(sub2ind(size(net_output), actions, 1:length(actions))), 'CBT');   % �������еȼ�
            % �����¸�״̬�����Qֵ
            next_q_values = forward(target_qnet, next_states);
            max_next_q_values = max(next_q_values);
            % ʱ�������Ŀ��
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
            
            % ʹ�� dlfeval ���ٱ�����������ʧ
            [dqn_loss, gradients] = dlfeval(@self.modelLoss, self.qnet, self.target_qnet, states, actions, next_states, rewards, dones);
            fprintf('LOSS: %f  water lv: %f  target lv: %f\n', dqn_loss, self.H, self.target_H);
            % ��������
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