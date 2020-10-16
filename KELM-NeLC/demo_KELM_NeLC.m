%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is an examplar file on how the KELM-NeLC [1] program could be used.
%
% [1] CHENG Yu-sheng, ZHAO Da-wei, WANG Yi-bin, PEI Gen-sheng.
%     Multi-label Learning of Kernel Extreme Learning Machine with Non-Equilibrium Label Completion. 
%     Acta Electronica Sinica, 2019, 47(3): 719-725.
%
% Please feel free to contact me (zhaodwahu@163.com), if you have any problem about this programme.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc
%% load data
addpath(genpath('.'));
load('emotion.mat')

%% set parameter
s=1;            %Suggest set to 1. Smoothing parameter.
alpha=0.5;      %Suggest set to [0.1-0.5]. Non-equilibrium parameter.
C=1;            %Suggest set to 1.This is a regularization parameter.
kernel_para=1.0;%Suggest set to 1.This is a kernel parpameter.
kernel_type='RBF_kernel';%Suggest set to 'RBF_kernel'.This is a kernel type.

%% the non-equilibrium label completion matrix construction
Conf= NeLC(train_target,alpha,s);
newtrain_target=Conf'*train_target;

[result,Outputs,Pre_Labels] = elm_kernel(test_data,test_target,train_data,newtrain_target,C,kernel_type,kernel_para);
