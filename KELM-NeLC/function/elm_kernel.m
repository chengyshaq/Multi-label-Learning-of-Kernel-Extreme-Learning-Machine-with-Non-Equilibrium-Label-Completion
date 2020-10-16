function [result,Outputs,Pre_Labels] = elm_kernel(test_data,test_target,train_data,train_target,C,kernel_type,kernel_para)
%%
% This function is designed to use the kernel-elm as classification. 
%
%   Syntax
%
%   INPUT:  test_data         - testing sample features, N_test-by-D matrix.
%           train_data        - training sample features, N-by-D matrix.
%           train_target      - training sample labels, l-by-N row vector.
%           C                 - this is a regularization parameter.
%           kernel_type       - this is a kernel type.
%           kernel_para       - this is a kernel parpameter.
%   OUTPUT: result            - this is a result of evaluation indexes.
%           Pre_Labels        - predicted labels, num_label-by-N_test row vector.
%           Outputs           - L x num_test data matrix of scores
%%
    [num_class,num_testing] = size(test_target);

    n = size(train_target,2);

    Omega_train = kernel_matrix(train_data,kernel_type,kernel_para);
    OutputWeight=((Omega_train+speye(n)/C)\(train_target')); %%Calculate the OutputWeight for predict

    Y=(Omega_train * OutputWeight)'; %   Y: the actual output of the training data

%%Calculate the output of testing input

    Omega_test = kernel_matrix(train_data,kernel_type,kernel_para,test_data);
    TY=(Omega_test'*OutputWeight)';    %   TY: the actual output of the testing data

%%%%%%%%%%Predict Labels
Outputs = TY;

Pre_Labels=zeros(num_class,num_testing);
for i=1:num_testing
    for j=1:num_class
        if(Outputs(j,i)>=0)
            Pre_Labels(j,i)=1;
        else
            Pre_Labels(j,i)=-1;
        end
    end
end
result.HL=Hamming_loss(Pre_Labels,test_target);
result.RL=Ranking_loss(Outputs,test_target);
result.OE=One_error(Outputs,test_target);
result.CV=coverage(Outputs,test_target);
result.AP=Average_precision(Outputs,test_target);
   





