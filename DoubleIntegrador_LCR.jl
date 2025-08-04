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
nx = 2; #Number of states or rows of the matrix A
nu = 1; #Number of inputs or columns of the matrix B 
ny = 2; #Number of output or rows of the matrix C

rx = 4; #Number of rows of the matrix X 
ru = 2; #Number of rows of the matrix U

A = [1 1; 0 1]
B = [2;1]
C = [1 0;0 1]

X = [0.8 0; -1 0; 0 1; 0 -1]
phi = ones(rx)
U = [1.2; -1.5]
varphi = ones(ru)


#Definition of the decision variables
@variable(model, K[1:nu,1:ny])
@variable(model, 0 <= H[1:rx,1:rx])
@variable(model, 0 <= M[1:ru,1:rx])
@variable(model, 0 <= lambda <= 0.999)

#Objective function
ppi = 0.5; #Choosen wheighting factor \pi
@objective(model, Min,  lambda)

@constraints(model, begin
#Positive invariance condition
H*X == X*(A+B*K*C);
H*phi <= lambda*phi;

#Control constraint
M*X == U*K*C;
M*phi <= varphi;
end )

print(model) #Print optimization problem
optimize!(model) #Solve optmization problem
@show objective_value(model) #Print objective function value
print(solution_summary(model)) #Prints the full solution text provided by the Knitro solver

#Saving the values of the variables
H = value.(H)
K = value.(K)
M = value.(M)
lambda = value.(lambda)

#Exporting solutions
writedlm("H.csv", H,',')
writedlm("K.csv", K,',')
writedlm("M.csv", M,',')
writedlm("lambda.csv", lambda,',')

#Exporting the full solution provided by the Knitro solver 
open("results.txt", "a") do io
  print(io, solution_summary(model))
close(io)
end
