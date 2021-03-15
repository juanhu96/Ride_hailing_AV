% Solver-Based Nonlinear Optimization for Ride-hailing AVs
% Jingyuan Hu, 2021-Mar-12

n = 4; % number of nodes
c = 1; % unit reallocation cost
gamma = 0.5; % share
lambda = [8;1;2;2]; % demand
r = [1;1;1;1]; % price
mu = [1;3;2;1]; % initial supply of HV
alpha = [1;2;2;1]; % initial supply of AV
d = [0 1 2 3 1 0 1 2 2 1 0 1 3 2 1 0];

x0 = [zeros(2*n*n,1);mu;alpha];
for i = 1:n
    x0(n*(i-1) + i) = mu(i);
    x0(n*n+n*(i-1) + i) = alpha(i);
end

%%%%%%%%% Linear Equality Constraints %%%%%%%%%

% supply before (sum across cols)
A_init = zeros(n,2*n^2+2*n);
for i = 1:n
    A_init(i, (i-1)*n+1:i*n) = ones(1,4);
end
A_init = [A_init; A_init];

% supply after (sum across rows)
A_after = diag(ones(1,4));
for i = 1:(n-1)
    A_after = [A_after diag(ones(1,4))];
end
A_after = [A_after zeros(n,n^2) diag(-ones(1,n)) zeros(n,n); zeros(n,n^2) A_after zeros(n,n) diag(-ones(1,n))];

Aeq = [A_init; A_after];
beq = [mu; alpha; zeros(n,1); zeros(n,1)];
lb = zeros(1,2*n^2+2*n);

%%%%%%%%% Linear Inequality Constraints %%%%%%%%%
A = [];
b = [];

%%%%%%%%% Nonlinear Constraint %%%%%%%%%

% function H(x), which is the optimal value of a LP
function lp_result = h(z)
global r gamma lambda
% construct the vector f from z
f = zeros(1, n);
for i = 1:n
    for j = 1:n
        f((i-1)*n + j) = gamma * r(j) * min(1, lambda(j) / z(2*n*n+j)+ z(2*n*n+n+j))...
            - d((i-1)*n + j);
Aeq = ; % matrix C
beq = ; % vector d
% return the optimal solution and optimal value
[x, fval] = linprog(f, Aeq, beq, lb); 
end

%%%%%%%%% Objective Function %%%%%%%%%

% s_mu = x(2*n*n+1:2*n*n+n)
% s_alpha = x(2*n*n+n+1:2*n*n+2n)
% g = x(n*n+1, 2n*n)
% profit HV: (1-gamma) * r.' * min(lambda, s_mu + s_alpha) * s_mu ./ (s_mu + s_alpha)
% profit AV: r.' * min(lambda, s_mu + s_alpha) * s_alpha ./ (s_mu + s_alpha)
% cost: c * d.' * g

fun = @(x) -sum((1-gamma) * r.' * min(lambda, x(2*n*n+1:2*n*n+n) + x(2*n*n+n+1:2*n*n+2*n))...
    * x(2*n*n+1:2*n*n+n) ./ (x(2*n*n+1:2*n*n+n) + x(2*n*n+n+1:2*n*n+2*n)) ...
    + r.' * min(lambda, x(2*n*n+1:2*n*n+n) + x(2*n*n+n+1:2*n*n+2*n)) ...
    * x(2*n*n+n+1:2*n*n+2*n) ./ (x(2*n*n+1:2*n*n+n) + x(2*n*n+n+1:2*n*n+2*n))...
    - c * d * x(n*n+1:2*n*n));

%%%%%%%%% Solve Optimization Problem %%%%%%%%%
opt = optimset('fmincon');
opt.algorithm = 'active-set';
% min -pi(x), then the optimal value is given by -fval
[x, fval] = fmincon(fun, x0, A, b, Aeq, beq, lb);


