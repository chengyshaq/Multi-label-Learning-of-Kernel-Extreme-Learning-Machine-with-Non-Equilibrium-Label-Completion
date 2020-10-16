function Conf= NeLC(target,alpha,s)
%%
% This function is designed to learn the non-equilibrium label completion matrix. 
% Note: The method of calculating the labels confidence matrix in the function uses the traditional information entropy.
%   Syntax
%
%   INPUT:  train_data        - training sample features, N-by-D matrix.
%           train_target      - training sample labels, l-by-N row vector.
%           alpha             - non-equilibrium parameter.
%           s                 - smoothing parameter.
%   OUTPUT: Conf              - L x L matrix of non-equilibrium label completion.
%%
    [num_class,num_training]=size(target);
%     [num_training,~]=size(train_data);
    a=zeros(num_class);
    b=zeros(num_class);
    P=zeros(num_class);
    N=zeros(num_class);
    [idx1,line1]=find(target==1);
    [idx0,line0]=find(target==-1);
    for i=1:num_class

        [u1]=find(idx1==i);
        [u0]=find(idx0==i);
        A{i}=line1(u1,1);
        B{i}=line0(u0,1);
    end
    for i=1:num_class
        for t=1:num_class
            a(i,t)=length(intersect(A{i},A{t}));%%在A中找1
            b(i,t)=length(intersect(B{i},A{t}));%%在A中找-1
            P(i,t)=-1/(((a(i,t)+s)/(num_training+s))*log2(((a(i,t)+s)/(length(find(target(i,:)==1))+s))));
            N(i,t)=-1/(((b(i,t)+s)/(num_training+s))*log2(((b(i,t)+s)/(length(find(target(i,:)==-1))+s))));
        end
    end
    
    P(logical(eye(size(P))))=0;
    N(logical(eye(size(N))))=0;
    
    T_A = P./repmat(sum(P,2),1,size(P,2));
    T_B = N./repmat(sum(N,2),1,size(N,2));

    Conf=(-T_A.*alpha+T_B.*(1-alpha));

    Conf=Conf-diag(diag(Conf))+diag(ones(size(Conf,1),1));
    Conf(isnan(Conf))=0;
end




