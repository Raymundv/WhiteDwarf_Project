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
using LaTeXStrings
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


#------------------Adding moderate noise to data:--------------------#
#--------------------------------------------------------------------#

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

solutionarray=Array(sol)
etasteps2=etasteps[1:end-60]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-60]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-60]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
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
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.1), callback=callback, maxiters = 200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\losses_moderate_noise2I.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer

p_trained = (layer_1 = (weight = [-1.567316271195498 -0.9216318933955566; -1.787292503457178 -0.6759406386557488; -0.6948343426553105 -0.4606772896221909; -2.2529477868387526 1.2214351198991988; -1.8678628735057883 -1.4683904157625742], bias = [-1.66385435414385; -1.7519604708403422; -1.6155886004415354; -1.6258139480569476; -1.2358590417683757]), layer_2 = (weight = [-1.2577146982545682 -0.6886468417225953 -2.2296825614802764 -1.6241713943766756 -1.3806466954583987; 2.106787908921441 1.4142801633102768 1.364842473555346 0.6464295506728144 1.9500866671569925; 1.211588400496942 1.427461371125084 -0.6322224289263189 0.6888368910955939 -0.14221225599731882; -0.7468674026845075 -0.06873937677288999 -0.6099932758781293 -0.20837492784033665 -0.4873153203685807; 0.4193302292982903 0.7733409309595426 -0.11620908447171212 0.05695189927426096 -0.07058877093245552], bias = [-1.0978714351152061; 1.4955956637014896; 0.19046756060855302; -0.19649180255284787; -0.21063014298132174]), layer_3 = (weight = [1.9238819615537468 1.5081552279569188 1.6857913153272854 2.102489150031562 1.062600073899889; -2.0775802195881328 -1.5616437576833242 -2.5821669237137974 -2.1897870856799075 -2.75393038562553; -2.920347665753938 -2.260786601118151 -1.865365792862753 -2.63266271642411 -2.9225201641932967; 1.839176241369884 0.8564540181954057 1.2384098663731784 1.059893843118218 1.9409821165077408; -0.9889823917105702 -1.5340086200913516 -1.5511629966592586 -1.5641214671299077 -1.4287503333442721], bias = [1.6035740673430354; -2.070902050430901; -2.0360506559737743; 1.2958467610133086; -1.439479423501944]), layer_4 = (weight = [-0.09288956951583764 -1.3926137511093306 -0.8317325507688428 0.4148214831055747 -0.5630208587764481; -0.45510074166063463 -1.3439348241423268 -0.948542004054094 -0.20214259722159883 -0.48427590211991206], bias = [-0.0019758379558647084; -0.548973815129146]))

# defining the time span for the plot

p = p_trained
#Retrieving the Data predicted for the Lotka Volterra model, with the UDE with the trained parameters for the NN
X̂ = predict_ude(p_trained)

# Plot the UDE approximation for  the CWDE model
pl_trajectory = scatter(etasteps2, transpose(X̂),markeralpha=0.4, xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth noisy data 
scatter!(etasteps2, transpose(training_array),title="Trained UDE vs Noisy Data", color = :black,markeralpha=0.4, label = ["Noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\UDE_trainedvsData_moderate_noiseI.png")


#--------------------forecasting---------------------#
#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#
function recovered_dynamics!(du,u,p,eta)
    phi, phiderivative = u
    output, _ = U([phi,phiderivative],p_trained,st)
    du[1] = output[1]+phiderivative
    du[2] = -2*phiderivative/eta+output[2]

    #output, _ = dudt2([phi,phiderivative],p,st)

    
end


open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Trained_parameters\\p_minimized_highnoise.txt","w") do f

    write(f, string(res1.minimizer))
end


#UDE prediction
prob_node_extrapolate = ODEProblem(recovered_dynamics!,I, etaspan,p)
_sol_node = solve(prob_node_extrapolate, Tsit5(),abstol=1e-15, reltol=1e-15,saveat = etasteps)

#UDE Extrapolation scatter plot
predicted_ude_plot = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
#UDE trained against training data
pl_trajectory = plot!(etasteps2, transpose(X̂), xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\trainedUDE90points_vsforecasted_udeI.png")




# Producing a scatter plot for the ground truth noisy data 
scatter(etasteps,transpose(x1_noise), color = :blue,markeralpha=0.5, label = ["Ground truth Noisy data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.2,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\UDE_Forecasted_vsNoisy_groundtruth_dataI.png")

# Producing a scatter plot for the ground truth ODE data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.5,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\UDE_Forecasted_vsODE_groundtruth_dataI.png")


#Final plot for the results- better formated
scatter(sol.t[1:end-60],Array(x1_noise[:,1:end-60])[1,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t[1:end-60],Array(x1_noise[:,1:end-60])[2,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi NODE
scatter!(collect(etasteps[1:end-60]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps[1:end-60]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-59:end],_sol_node[1,end-59:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-59:end],_sol_node[2, end-59:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\Whitedwarf_forecasted_modelUDE.png")

#Second version



#plot!(_sol_node)


#final actual version 

scatter(sol.t[1:end-60],Array(x1_noise[:,1:end-60])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-59:end],Array(x1_noise[:,41:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(eta, X̂[1,:],color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-60:end],_sol_node[1,end-60:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\NeuralODEModel_finalversion.png")


# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:40],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-60:end],p_trained,st)[1]

plot!(sol.t[40:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.30, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :bottomright)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\HighNoise\\Recoveredterm2_nonoise.png")




