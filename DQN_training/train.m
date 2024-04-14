clear all;
clc;

abspath=utils('abspath');
policy=DQN(false);
replay_buffer = ReplayBuffer(20000);
for i = 1:1:50
    fprintf("epoch = %d", i);
    env = Env(abspath('sys.ini'));
    if (env.succeed)
        observation = env.reset();
        state = observation.agent.H;
        while 1
%             env.render();
            [u, action]=policy.action(observation);
    
            [next_observation,done,info]=env.step(action);
            next_state = next_observation.agent.H;

            replay_buffer.add([state observation.targetHeight], u, -abs(next_observation.agent.H - next_observation.targetHeight), [next_state next_observation.targetHeight], done);
    
            state = next_state;
            if replay_buffer.size > policy.n_update_start             % 当经验重放池有policy.n_update_start个数据之后，再进行更新
                [states, u_action, rewards, next_states, dones] = replay_buffer.sample(policy.batch_size);        % policy.batch_size是单次训练batch size，越大模型收敛越快，但对cpu性能要求越高
                policy.update(states, u_action, rewards, next_states, dones);
            end
            disp(info);
            if(done)
                break;
            end
            wait(50);
        end
    end
    net = policy.get_qnet();
    output_path = sprintf('DQN_training/network_weights_bc%d_naction%d_fc%d.mat', policy.batch_size, policy.n_action, policy.n_hidden_neurual);
    save(output_path, 'net');

end



function wait (ms)
time=ms/1000;
% tic
pause(time)
% toc
end

