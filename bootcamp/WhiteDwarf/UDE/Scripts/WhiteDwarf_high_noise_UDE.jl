#Loading the necessary libraries
using Plots
using DifferentialEquations
using Random
using Statistics
using OrdinaryDiffEq
using Lux 
using DiffEqFlux
using ComponentArrays 
using Optimization, OptimizationOptimJL,OptimizationOptimisers   
using JLD
using OptimizationFlux

using Statistics                                                                
rng = Random.default_rng()
Random.seed!(99)

#Constants
C = 0.01


#Initial Conditions
I = [1.0, 0.0]   #Psi(0)=1, Psi'(0)=1
etaspan = (0.05, 5.325)

#radius range
datasize= 100
etasteps = range(etaspan[1], etaspan[2]; length = datasize)

#Define the whitedwarf equation as a function
function whitedwarf(du, u, p, r)
    psi = u[1]
    dpsi = u[2]
    du[1] = dpsi
    du[2] = (-((psi^2-C))^(3/2) - 2/r * dpsi)
end


#Defining the Ordinary differential equation as an ODEProblem with the DifferentialEquations.jl
prob = ODEProblem(whitedwarf, I, etaspan)
#Solving the ODEProblem with the Tsit5() algorithm
sol = solve(prob,saveat=etasteps)
eta=sol.t
#Plot
plot(sol, linewidth = 1, title = "White Dwarf equation", xaxis = "\\eta",
     label = ["\\phi" "\\phi'"])

#--------------I will solve the white dwarf equation using the SecondOrderODEProblem function------------

#Defining the function containing the Second Order Differential Equation
function whitedwarf2(ddu,du,u,C,eta)
    ddu .= (-((u.*u.-C)).^(3/2) - 2/eta * du)
end

#Initial conditions definined as required by the syntax of the Second Order Differential Equation
dpsi0=[0.0]
psi0=[1.0]
#Defining the secondOrderProblem 
prob2 = SecondOrderODEProblem(whitedwarf2,dpsi0, psi0, etaspan, C)
#Solving it with the automated choosen algorithm
sol2 = solve(prob2, saveat=etasteps)

#plot sol2
plot(sol2, linewidth=1.5, title = "White Dwarf equation", xaxis = "\\eta", label = ["\\phi" "\\phi '"])

x1=Array(sol)

x1_mean = mean(x1, dims = 2)
noise_magnitude = 35e-2
x1_noise = x1 .+ (noise_magnitude*x1) .* randn(eltype(x1), size(x1))
#Displaying true data vs noisy data
plot(sol, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(sol.t, transpose(x1_noise), color = :red, label = ["Noisy Data" nothing])


#------------------------Defining the UDE ---------------------#
#---------------------Defining the neural network.-------------------

# Gaussian RBF as the activation function for the Neurons.
rbf(x) = exp.(-(x.^2))

# Neural Network structure
U = Lux.Chain(
    Lux.Dense(2,5,rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,2)
)

# Get the initial parameters and state variables of the model (Setting up the initial parameters for the NN)
p, st = Lux.setup(rng, U)

# Defining the model with the NN approximation for the neural network UDE.
function ude_dynamics(du,u, p, eta)
   NN = U(u, p, st)[1] # Network prediction
   du[1] = u[2] + NN[1]
   du[2] = -2*u[2]/eta + NN[2]
end


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan, p)

#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)

function predict(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end


# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict(theta)
    sum(abs2, x1_noise .- X̂)
end

# Defining an empty list to store the losses throughout the training process 
losses = Float64[]

# Defining the callback function
callback = function (p, l)
  push!(losses, l)
  if length(losses)%50==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  return false
end

##------------------ Training the UDE with the ground truth data -------------------------#
##------------------------------------------------------------------------------##



#Setting up the optimization process
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)

#Training with ADAM.
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res = Optimization.solve(optprob, ADAM(0.2), callback=callback, maxiters = 300)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
#Refined training with BFGS

optprob1 = Optimization.OptimizationProblem(optf, res.minimizer)
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.006), callback=callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\losses_high_noise2.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer



open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Trained_parameters\\p_minimized_highnoise2.txt","w") do f

    write(f, string(res1.minimizer))
end


p_trained = (layer_1 = (weight = [-0.8901191381367268 -4.712719172855476; -1.395617491250587 -1.804574210577113; -0.7344898093032929 -1.942212612334039; -0.44711712479711585 1.023016097056906; -6.625656609687567 -6.684580293067858], bias = [-0.10833792597901784; -1.769100133641662; -1.1923323305783946; -1.9091544455156515; 1.2625439221333146]), layer_2 = (weight = [-1.5793628368178416 -0.8415109049193342 -2.3706479701641006 -1.9657528990633941 -1.7420865925877995; 2.153021509563261 0.2847678614209927 2.5421532820478694 -0.11720926929394745 8.059755757015022; 1.8300323736445845 1.9221380055463675 1.3961313052140738 1.5241758902254563 0.8989351178156105; -2.029645018736675 -1.0120032518660862 -1.6057072346046715 -1.4042035900852774 -1.2243428990077976; 1.3372884198747854 1.6754062589766479 1.521857091120925 1.100357924551906 1.2934847603823685], bias = [-1.9242574764125826; -1.707623590699739; 1.8129137779089604; -1.8227284775018369; 2.0260013649889337]), layer_3 = (weight = [1.821843677673886 10.85960612013984 1.5819958735703457 1.7055252065097175 0.5479163886500312; 1.6794059250095674 2.2075552500439084 1.426583728931246 1.3915263313295299 0.7973136477407702; -2.372429818236951 -0.9366538781345051 -1.1009851383555125 -2.190365554650455 -2.014984043703852; 1.725216737603289 -8.5400209634787 0.4633640732735316 1.0450346581383982 1.1765699671533703; -0.9033007189119121 0.517238968630953 -1.470613898684292 -1.2788883397688258 -1.0306158273718942], bias = [0.01973765031904054; 4.620851709653557; -2.021850740118817; 4.0415154372370194; -0.15343519801237002]), layer_4 = (weight = [-0.8349917219432856 -0.4320754065343898 -0.5881330874267477 -6.037880504055824 -7.757974329588875; 1.0479067963597841 -0.5490514005973874 0.37853753268092516 2.547059291338586 7.99070463241802], bias = [8.379970532777392; -8.882872523303774]))

#Retrieving the Data predicted for the Lotka Volterra model, with the UDE with the trained parameters for the NN
X̂ = predict(p_trained, I, etasteps)

# Plot the UDE approximation for  the Lotka Volterra model
pl_trajectory = plot(etasteps, transpose(X̂),title="Trained UDE", xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth data 
scatter!(sol.t, transpose(x1_noise), color = :black,markeralpha=0.4, label = ["Ground truth noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\UDEvsODE_high_noise2.png")
#PRevious version was with  Optim.BFGS(initial_stepnorm=0.01)
#Previous version




#Final plot for the preprint 
#Last Version for the preprint

#----------------------------------




scatter(sol.t,Array(x1_noise[:,1:end])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t,Array(x1_noise[:,1:end])[2,:],color=:blue,markeralpha=0.3, linewidth = 1,markershape=:diamond, xaxis = "\\eta",
     label = "Training \\phi' ", title="Trained UDE")


#scatter!(sol.t[1:end],Array(sol[:,1:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(sol.t[1:end],X̂[1, :],color=:black,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")




plot!(sol.t[end-99:end],X̂[2, :],color=:black,linestyle=:dash,label="Predicted \\phi'")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\NeuralODEModel_finalversion_highnoise.png")



#Recovering missing term 
# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t,Y_guessed[2,:], label = "UDE Approximation")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end


Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.35, label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :right)

title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\Recoveredterm2_highnoise.png")

