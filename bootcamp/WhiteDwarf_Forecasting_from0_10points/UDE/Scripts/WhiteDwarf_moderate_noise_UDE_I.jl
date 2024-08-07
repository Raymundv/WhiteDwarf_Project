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
noise_magnitude = 7e-2
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
etasteps2=etasteps[1:end-90]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-90]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-90]

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
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.001), callback=callback, maxiters = 1500)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\losses_moderate_noise2I.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer

p_trained = (layer_1 = (weight = [5.077156451573874 7.137186877510023; 1.8018128269842208 -0.5975681071842939; 0.8519706774409836 1.7512322234756346; -1.8004716402255796 -1.6809355228465797; -3.2541323424284236 -1.9648136129151026], bias = [0.030246795219938748; 1.8045822813676136; -1.3456298625998744; -1.051214065363055; -2.3807676163214686]), layer_2 = (weight = [-1.8092651197156537 -1.1063215707124705 -2.4751831476204886 -1.8936194269134472 -1.7430736145632457; 0.8617384799842778 1.2314361001476135 2.8345043589008774 0.5825637300584665 1.7672063496060937; 15.558547315822146 1.4870697397937098 8.638195792232912 0.9988500617951418 0.3184167394705994; 51.33275307951912 0.8265486152136737 -67.44646562278776 0.38208144754723117 -0.9281517012848063; 2.7551928642617627 1.2255328799034124 1.890079679410556 0.8037435463799448 0.8825754466086163], bias = [-2.186189967374538; -0.48733228245635724; 14.696334693057645; 45.733165201096284; 1.7207807312411143]), layer_3 = (weight = [2.13531389282426 1.5602158074828065 1.7114237003063804 2.459595874344665 1.2656095797858768; -1.7197287811009279 -1.9867810745242869 -1.7736394042897168 -3.069762507121461 -2.6152861690527582; -2.7287590638747847 -1.8518809688531948 -1.5709499096256583 -2.8599403500460294 -2.6062525127432004; 1.7093486660367645 3.2145116415728614 -3.7611551883330314 7.675961220033988 1.2545356000769583; -1.1123201820843633 -1.7240627190221187 -1.2390286487571278 -3.3297546105548204 -1.5793349497634346], bias = [4.552018972616008; -1.850688604908136; -0.43801099107958275; 1.151037288385348; -5.811776472373979]), layer_4 = (weight = [-0.33966086701221654 -2.20983308674166 -2.213073103076914 -34.76811413064417 -2.311776195931363; -0.34638670418974077 -1.35029428292262 -0.2087780745420773 14.032812110790971 1.396733084684465], bias = [0.8715646877432137; -1.0012920777755]))
p = p_trained
# defining the time span for the plot
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Trained_parameters\\p_minimized_moderatenoise.txt","w") do f

    write(f, string(p_trained))
end

#Retrieving the Data predicted for the White Dwarf model, with the UDE with the trained parameters for the NN
X̂ = predict_ude(p_trained)

# Plot the UDE approximation for  the White Dwarf model
pl_trajectory = scatter(etasteps2, transpose(X̂),markeralpha=0.4, xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth noisy data 
scatter!(etasteps2, transpose(training_array),title="Trained UDE vs Noisy Data", color = :black,markeralpha=0.4, label = ["Noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\UDE_trainedvsData_moderate_noiseI.png")


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



#UDE prediction
prob_node_extrapolate = ODEProblem(recovered_dynamics!,I, etaspan,p_trained)
_sol_node = solve(prob_node_extrapolate, Tsit5(),abstol=1e-15, reltol=1e-15,saveat = etasteps)

#UDE Extrapolation scatter plot
predicted_ude_plot = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
#UDE trained against training data
pl_trajectory = plot!(etasteps2, transpose(X̂), xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\trainedUDE90points_vsforecasted_udeI.png")




# Producing a scatter plot for the ground truth noisy data 
scatter(etasteps,transpose(x1_noise), color = :blue,markeralpha=0.5, label = ["Ground truth Noisy data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.2,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsNoisy_groundtruth_dataI.png")

# Producing a scatter plot for the ground truth ODE data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.5,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsODE_groundtruth_data.png")


#plot()

#Final plot for the results- better formated
scatter(sol.t[1:end-80],Array(x1_noise[:,1:end-80])[1,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t[1:end-80],Array(x1_noise[:,1:end-80])[2,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi UDE
scatter!(collect(etasteps[1:end-80]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps[1:end-80]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-79:end],_sol_node[1,end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-79:end],_sol_node[2, end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\Whitedwarf_forecasted_modelUDE.png")

#final actual version 

scatter(sol.t[1:end-90],Array(x1_noise[:,1:end-90])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-89:end],Array(x1_noise[:,11:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(eta, X̂[1,:],color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-90:end],_sol_node[1,end-90:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\NeuralODEModel_finalversion.png")


# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:10],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-90:end],p_trained,st)[1]

plot!(sol.t[10:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.25, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :topright)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_10points\\UDE\\Results\\ModerateNoise\\Recoveredterm2_nonoise.png")



