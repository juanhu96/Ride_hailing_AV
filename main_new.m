% Solver-Based Nonlinear Optimization for Ride-hailing AVs
% Jingyuan Hu, 2021-Mar-12

n = 4; % number of nodes
c = 2; % unit reallocation cost
gamma = 0.1; % share
r = [6;2;2;3]; % price
d = [0 1 2 3 1 0 1 2 2 1 0 1 3 2 1 0];

lambda = [8;1;2;4]; % demand
% mu = [1;3;2;1]; % initial supply of HV
% alpha = [1;2;2;1]; % initial supply of AV

mu = [2;5;3;1]; % initial supply of HV
alpha = [1;3;5;1]; % initial supply of AV

x0 = [zeros(2*n*n,1);mu;alpha];
for i = 1:n
    x0(n*(i-1) + i) = mu(i);
    x0(n*n+n*(i-1) + i) = alpha(i);
end

%%%%%%%%% Linear Equality Constraints %%%%%%%%%

% supply before (sum across cols)
A_init = zeros(n,n^2);
for i = 1:n
    A_init(i, (i-1)*n+1:i*n) = ones(1,n);
end
A_init = [A_init zeros(n, n^2) zeros(n, 2*n); zeros(n, n^2) A_init zeros(n, 2*n)];

% supply after (sum across rows)
A_after = diag(ones(1,n));
for i = 1:(n-1)
    A_after = [A_after diag(ones(1,n))];
end
A_after = [A_after zeros(n,n^2) diag(-ones(1,n)) zeros(n,n); zeros(n,n^2) A_after zeros(n,n) diag(-ones(1,n))];

Aeq = [A_init; A_after];
beq = [mu; alpha; zeros(n,1); zeros(n,1)];

%%%%%%%%% Linear Inequality Constraints %%%%%%%%%
A = [];
b = [];

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
options = optimoptions('fmincon','Display','iter-detailed');
% min -pi(x), then the optimal value is given by -fval
ub = [];
% for the interior point method, 
% the iterations must lie in the interior lb < x < ub.
lb = zeros(size(x0)); % permit x to be negative for searching
nonlcon = @h;
[x, fval] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);

%%%%%%%%% Nonlinear Constraint %%%%%%%%%

% function H(x), which is the optimal value of a LP
function [c,ceq] = h(z)
% since this is defined before the constant is defined
% need to define it directly inside the function instead of using global
n = 4; % number of nodes
c = 1; % unit reallocation cost
gamma = 0.5; % share
lambda = [8;1;2;2]; % demand
r = [1;1;1;1]; % price
mu = [1;3;2;1]; % initial supply of HV
alpha = [1;2;2;1]; % initial supply of AV
d = [0 1 2 3 1 0 1 2 2 1 0 1 3 2 1 0];

% construct the vector f from z
f = zeros(1, n*n);
for i = 1:n
    for j = 1:n
        f((i-1)*n + j) = gamma * r(j) * min(1, lambda(j) / z(2*n*n+j)+ z(2*n*n+n+j))...
            - d((i-1)*n + j);
    end
end

C_init = zeros(n,n*n);
for i = 1:n
    C_init(i, (i-1)*n+1:i*n) = ones(1,n);
end
C_after = diag(ones(1,n));
for i = 1:(n-1)
    C_after = [C_after diag(ones(1,n))];
end

Aeq = [C_init; C_after]; % matrix C
beq = [mu; z(2*n*n+1:2*n*n+n)]; % vector d
A = [];
b = [];
lb = zeros(n*n,1);
% lb = -ones(n*n,1);
ub = [];

% return the optimal solution and optimal value
[x, fval, output] = linprog(f, A, b, Aeq, beq, lb, ub, optimoptions('linprog','Display','none'));
c = -fval;
ceq = [];
end
