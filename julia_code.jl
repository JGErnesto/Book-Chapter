#Package imports
using LinearAlgebra
using CSV, DelimitedFiles
using KNITRO
using JuMP, NEOSServer

#Neos configuration
model = Model() do
    return NEOSServer.Optimizer(; email = "your_email@domain", solver = "Knitro")
end

#Solver configuration
set_optimizer_attribute(model, "ms_enable", 1); #Multistart
set_optimizer_attribute(model, "ms_maxsolves", 200); 

# System data - Quasi-LPV system
A = zeros(2,2,2)
A[:,:,1] = [1 -1; -1 -0.5];
A[:,:,2] = [1 1; -1 -0.5];

B = zeros(2,1,2);
B[:,:,1] = [6;2];
B[:,:,2] = [4; -2];

C = [1 0];

# State, control and control-rate (matrix) constraints
X = [1 0; -1 0; 0 0; 0 0];
U = [1; -1.25];
Udelta = [5; -5];

#  System and constraint parameters
nx = 2; # Number of states or rows of the matrix A
nu = 1; # Number of inputs or columns of the matrix B 
ny = 1; # Number of outputs or rows of the matrix C
nv = 2; # Total number of vertices of the politope [A(\alpha_k) B(alpha_k)]
rdelta = 2; #Number of rows of the matrix U_delta
rx = 4; # Number of rows of the matrix X 
ru = 2; # Number of rows of the matrix U

# Design parameters
t = 8; # Number of directions, \bar{t}
psix = [1 0 -1 0 1 1 -1 -1; 0 1 0 -1 1 -1 1 -1]  # Chosen directions \Psi_{x} 

# Auxiliary bound parameter
bGamma = 100;

# Definition of the decision variables and bounds
@variable(model, -bGamma <= psiu[1:nu, 1:t] <= bGamma) 

@variable(model, 0 <= H[1:rl, 1:rl, 1:nv, 1:nv] <= bGamma)
@variable(model, -bGamma <= Lx[1:rl,1:nx] <= bGamma)
@variable(model, -bGamma <= Lu[1:rl,1:nu] <= bGamma)

@variable(model, -bGamma <= K[1:nu, 1:ny,1:nv,] <= bGamma)
@variable(model, -bGamma <= bK[1:nu, 1:nu, 1:nv] <= bGamma)
@variable(model, -bGamma <= hK[1:nu, 1:ny, 1:nv] <= bGamma)
#hK = zeros(nu,ny,nv); # To ensure hK = 0, use this line instead of previous one

@variable(model, -10*bGamma <= J[1:(nx+nu),1:rl] <= 10*bGamma)
@variable(model, 0 <= G[1:(rx+ru),1:rl] <= bGamma)
@variable(model, 0 <= Q[1:rdelta, 1:rl, 1:nv, 1:nv] <= bGamma)
@variable(model, 0 <= gamma[1:t] <= bGamma)
@variable(model, 0 <= lambda <= 0.999)

#
# BP design for LPV systems
#

#Objective function of BP (44)
ppi = 0.5; # Chosen weighting factor \pi

@objective(model, Min, (ppi)*lambda - (1-ppi)*sum(gamma)/t)

# Optimization Constraints
for i = 1:nv
    for j= 1:nv

@constraints(model, begin
#Positive Invariance conditions (36a)-(36b)
H[:,:,i,j]*[Lx Lu] == [Lx Lu]*[A[:,:,i] B[:,:,i]; K[:,:,i]*C+hK[:,:,j]*C*A[:,:,i] I(nu)+bK[:,:,i]+hK[:,:,j]*C*B[:,:,i]];
H[:,:,i,j]*ones(rl) <= lambda*ones(rl);

#Control rate constraints fulfillment through (38a)-(38b)
Q[:,:,i,j]*[Lx Lu] == Udelta*[K[:,:,i]*C + hK[:,:,j]*C*A[:,:,i]  bK[:,:,i] + hK[:,:,j]*C*B[:,:,i]];
Q[:,:,i,j]*ones(rl) <= ones(rdelta);
 end)
    end
end

@constraints(model, begin
# Respect of augmented state constraints through (37a)-(37b)
[X zeros(rx,nu); zeros(ru,nx) U] == G*[Lx Lu];
G*ones(rl) <= ones(rx+ru);

# Pseudo inverse condition (39)
J*[Lx Lu] == I(nx+nu);

# Directions inclusion conditions, equation (45) in a matrix form
Lx*psix*Diagonal(gamma) + Lu*psiu .<= ones(rl,t);
end)

# Solve optmization problem
optimize!(model) 

#Exporting information about the solution provided by the Knitro solver 
open("results.txt", "a") do io
  print(io, solution_summary(model))
close(io)
end

# Saving the values of the decision variables
H = value.(H)
Lx = value.(Lx)
Lu = value.(Lu)
K = value.(K)
bK = value.(bK)
hK = value.(hK)
G = value.(G)
Q = value.(Q)
J = value.(J)
lambda = value.(lambda)
gamma = value.(gamma)
psiu = value.(psiu)

# Exporting the saved values for external use
for i = 1:nv
    for j = 1:nv
writedlm("H$i$j.csv", H[:,:,i,j],',')
writedlm("Q$i$j.csv", Q[:,:,i,j],',')
    end
writedlm("K$i.csv", K[:,:,i],',')
writedlm("bK$i.csv", bK[:,:,i],',')
writedlm("hK$i.csv", hK[:,:,i],',')
end

writedlm("Lx.csv", Lx,',')
writedlm("Lu.csv", Lu,',')
writedlm("G.csv", G,',')
writedlm("J.csv", J,',')
writedlm("lambda.csv", lambda,',')
writedlm("gamma.csv", gamma,',')
writedlm("psix.csv", psix,',')
writedlm("psiu.csv", psiu,',')
writedlm("ppi.csv", ppi, ',')


