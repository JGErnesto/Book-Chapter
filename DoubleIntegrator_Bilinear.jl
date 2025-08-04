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


#Parameters

rl = 6; #Number of rows of the matrix L
t = 8; #Total number o directions, \bar{t}

A = [1 1; 0 1]
B = [2;1]
C = [1 0]

X = [0.8 0; -1 0; 0 1; 0 -1]
U = [1.2; -1.5]

nx = 2; #Number of states or rows of the matrix A
nu = 1; #Number of inputs or columns of the matrix B 
ny = 1; #Number of output or rows of the matrix C

rx = 4; #Number of rows of the matrix X 
ru = 2; #Number of rows of the matrix U


#Choosen directions
psix = [1 0 -1 0 1 1 -1 -1; 0 1 0 -1 1 -1 1 -1] #Choosen direction, \Psi_{x}

#Definition of the decision variables
@variable(model, 0 <= H[1:rl,1:rl] )
@variable(model, 0 <= M[1:ru,1:rl] )
@variable(model, 0 <= T[1:rx,1:rl] )
@variable(model, L[1:rl,1:nx] )
@variable(model, K[1:nu,1:ny] )
@variable(model, J[1:nx,1:rl] )
@variable(model, 0 <= lambda <= 0.999)
@variable(model, 0 <= gamma[1:t] )
@variable(model, psiu[1:nu, 1:t] )

#Objective function
ppi = 0.5; #Choosen wheighting factor \pi
@objective(model, Min,  (ppi)*lambda - (1-ppi)sum(gamma)/t)

#Constraints
@constraints(model, begin
#Positive Invariance Condition
H*L == L*(A+B*K*C);
H*ones(rl) <= lambda*ones(rl);

#State Constraints
T*L == X ;
T*ones(rl) <= ones(rx);

#Control Constraint
M*L == U*K*C;
M*ones(rl) <= ones(ru);

#Pseudo Inverse Condition
J*L == I(nx);

#Directions inclusion condition, matrix form
Lx*psix*Diagonal(gamma) + Lu*psiu .<= ones(rl,t);
end )

print(model) #Print optimization problem
optimize!(model) #Solve optmization problem
@show objective_value(model) #Print objective function value
print(solution_summary(model)) #Prints the full solution text provided by the Knitro solver

#Saving the values of the variables
H = value.(H)
L = value.(L)
K = value.(K)
M = value.(M)
T = value.(T)
J = value.(J)
lambda = value.(lambda)
gamma = value.(gamma)
psiu = value.(psiu)

#Exporting solutions
writedlm("H.csv", H,',')
writedlm("L.csv", L,',')
writedlm("K.csv", K,',')
writedlm("M.csv", M,',')
writedlm("T.csv", T,',')
writedlm("J.csv", J,',')
writedlm("lambda.csv", lambda,',')
writedlm("gamma.csv", gamma,',')
writedlm("psiu.csv", psiu,',')

#Exporting the full solution provided by the Knitro solver 
open("results.txt", "a") do io
  print(io, solution_summary(model))
close(io)
end
