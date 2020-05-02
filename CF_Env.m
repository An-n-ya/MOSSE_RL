%% CF_Env
clc
clear
ObservationInfo = rlNumericSpec([163 115]);  %为环境创造连续的action或observation
ObservationInfo.Name = 'CartPole States';

ActionInfo = rlFiniteSetSpec([0 1]);  %为环境创造离散的action或observation
ActionInfo.Name = 'CartPole Action';

[InitialObservation,LoggedSignals] = CF_Reset();

Action = 1;
[Observation,Reward,IsDone,LoggedSignals] = CF_Step(Action,LoggedSignals);

env = rlFunctionEnv(ObservationInfo,ActionInfo,'CF_Step','CF_Reset');%运行rlFunctionEnv会自动的调用validate程序

InitialObs = reset(env)

[NextObs,Reward,IsDone,LoggedSignals] = step(env,0);
NextObs