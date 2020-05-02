%% 创建预设环境
clc
clear
ObservationInfo = rlNumericSpec([163 115]);  %为环境创造连续的action或observation
ObservationInfo.Name = 'CartPole States';

ActionInfo = rlFiniteSetSpec([0 1]);  %为环境创造离散的action或observation
ActionInfo.Name = 'CartPole Action';
env = rlFunctionEnv(ObservationInfo,ActionInfo,'CF_Step','CF_Reset');

%% Create PG agent
%创建actor深度神经网络
actorNetwork = [
    imageInputLayer([163 115 1],'Normalization','none','Name','state')
    convolution2dLayer([5 5], 4, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 3)
    convolution2dLayer([3 3], 8, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(1, 'Stride',1)
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(2)
    softmaxLayer
    ];
%设置actor的学习速率以及梯度阈值
actorOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);
%创建actor representation
actor = rlStochasticActorRepresentation(actorNetwork,ObservationInfo,ActionInfo,'Observation',{'state'},actorOpts);
%使用默认选项创建actor
agent = rlPGAgent(actor);

%% Train Agent
%设置训练参数
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 243, ...
    'Verbose', false, ...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',200,...
    'ScoreAveragingWindowLength',20);
%绘制环境
% plot(env);
%训练模型
trainingStats = train(agent,env,trainOpts);
%保存模型
save(opt.SaveAgentDirectory + "/finalAgent.mat",'agent')