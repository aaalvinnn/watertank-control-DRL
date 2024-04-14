classdef ReplayBuffer < handle
    % 经验回放池
    properties
        buffer
        capacity
    end
    
    methods
        function obj = ReplayBuffer(capacity)
            obj.buffer = {};
            obj.capacity = capacity;
        end
        
        function add(obj, state, action, reward, next_state, done)
            if length(obj.buffer) >= obj.capacity
                obj.buffer(1) = [];
            end
            obj.buffer{end+1} = {state, action, reward, next_state, done};
        end
        
        function [state, action, reward, next_state, done] = sample(obj, batch_size)
            if batch_size > length(obj.buffer)
                error('Batch size exceeds buffer size');
            end
            indices = randperm(length(obj.buffer), batch_size);
            batch = obj.buffer(indices);
            batch_data = vertcat(batch{:});
            state = vertcat(batch_data{:,1})';
            action = vertcat(batch_data{:,2})';
            reward = vertcat(batch_data{:,3})';
            next_state = vertcat(batch_data{:,4})';
            done = vertcat(batch_data{:,5})';
        end
        
        function sz = size(obj)
            sz = length(obj.buffer);
        end
    end
end