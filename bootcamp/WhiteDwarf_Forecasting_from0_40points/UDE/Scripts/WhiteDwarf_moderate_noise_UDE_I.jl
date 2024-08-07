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
res = Optimization.solve(optprob, ADAM(0.1), callback=callback, maxiters = 300)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
#Refined training with BFGS

optprob1 = Optimization.OptimizationProblem(optf, res.minimizer)
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 1500)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\losses_moderate_noise2I.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer


#p_trained = (layer_1 = (weight = [-1.1710321106924275 2.5497610416841736; -1.9752445306473239 -0.4573892100916554; 1.6377064109128705 -1.6081172362584286; -0.7303985644010678 1.2028036295733968; -2.666469851294151 -4.158182508584599], bias = [2.4789480525469876; -0.5203304120726031; -2.58376495214217; -0.19987295628932777; 0.9522062182626337]), layer_2 = (weight = [-0.8223889075285493 -0.245671249126329 -1.5019329964930017 -1.2266193887311756 -0.9767458094840697; 1.5266945600451747 0.8437557491523062 0.3861403515122325 0.19182662415228566 1.4744900266471936; 0.09555174338142916 1.4141114949523585 -0.07886738972050077 -0.01627271170720193 -0.0059473274454871945; -1.8891967044796034 0.17495298568639125 -1.6858974814321348 0.7337557240870672 2.0636936415141816; 1.7107860225266769 -0.28055343033146524 1.824570138282683 -0.5198923476348057 -3.142815750210537], bias = [-0.8620208655134972; 0.8005607677591499; 0.038968763131035415; -0.5409118657528039; 1.6045735416625158]), layer_3 = (weight = [1.3524760726175225 0.8922443764754253 1.3349671999141608 1.550347702983861 0.5288505480719378; -0.34258069672908237 0.320373762561616 -0.5258174328844744 2.061129334557985 -1.7664012555659834; -1.639589106588196 -0.8413368436192847 -0.6579174170861408 -1.8328519239696226 -1.6189572585156096; 1.1910816393941492 0.15273681749379464 -0.3992892980379861 5.035946425021751 2.2925780609874855; -0.4110814092750676 -0.9001970490534819 -1.280353978291379 -3.0580640471003644 0.5174244646128212], bias = [1.6563112746096535; -0.7241227822065467; -1.5171664407406378; 1.1895957543590248; -1.9261569157353373]), layer_4 = (weight = [-0.2260154344518752 -0.5442263109488109 -0.11104108875844558 1.6676848697643798 3.621700523146915; -1.1583621164368048 -1.694296914253846 -0.2777702534383362 -5.970827822431849 -0.8876342761559407], bias = [0.21772962561970657; 0.2181806589011117]))
#p = p_trained 


# defining the time span for the plot
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Trained_parameters\\p_minimized_moderatenoise.txt","w") do f

    write(f, string(p_trained))
end

#Retrieving the Data predicted for the White Dwarf model, with the UDE with the trained parameters for the NN
X̂ = predict_ude(p_trained)

# Plot the UDE approximation for  the White Dwarf model
pl_trajectory = scatter(etasteps2, transpose(X̂),markeralpha=0.4, xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth noisy data 
scatter!(etasteps2, transpose(training_array),title="Trained UDE vs Noisy Data", color = :black,markeralpha=0.4, label = ["Noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\UDE_trainedvsData_moderate_noiseI.png")


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


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\trainedUDE90points_vsforecasted_udeI.png")




# Producing a scatter plot for the ground truth noisy data 
scatter(etasteps,transpose(x1_noise), color = :blue,markeralpha=0.5, label = ["Ground truth Noisy data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.2,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsNoisy_groundtruth_dataI.png")

# Producing a scatter plot for the ground truth ODE data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.5,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsODE_groundtruth_data.png")


#plot()

#Final plot for the results- better formated
scatter(sol.t[1:end-60],Array(x1_noise[:,1:end-60])[1,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t[1:end-60],Array(x1_noise[:,1:end-60])[2,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi UDE
scatter!(collect(etasteps[1:end-60]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps[1:end-60]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-59:end],_sol_node[1,end-59:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-59:end],_sol_node[2, end-59:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\Whitedwarf_forecasted_modelUDE.png")

#final actual version 

scatter(sol.t[1:end-60],Array(x1_noise[:,1:end-60])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-59:end],Array(x1_noise[:,41:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(eta, X̂[1,:],color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-60:end],_sol_node[1,end-60:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\NeuralODEModel_finalversion.png")


# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:40],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-60:end],p_trained,st)[1]

plot!(sol.t[40:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.35, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :right)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\ModerateNoise\\Recoveredterm2_nonoise.png")




