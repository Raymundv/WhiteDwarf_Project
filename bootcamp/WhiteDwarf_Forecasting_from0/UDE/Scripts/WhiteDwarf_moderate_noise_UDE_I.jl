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
etasteps2=etasteps[1:end-10]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-10]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-10]

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
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\losses_moderate_noise2I.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer



#p_trained = (layer_1 = (weight = [5.434005994995207 -2.5992722478801977; -9.57733591004127 -10.634827473142202; -0.9967072322479994 -1.3982282512018958; 7.147376942835671 4.679283095452256; -5.848607833623388 -16.477837822920335], bias = [-6.276257084737525; -1.135550712553739; 0.06728830421322665; 3.1886444680165953; -1.4781405560903216]), layer_2 = (weight = [-4.648768502784665 -0.1110103985478996 -10.822135400886754 -1.3256141561034285 -8.112330068062928; -1.880172158940449 -9.712409688097432 -2.8460938042711867 -8.64097552589408 -1.1708312641391887; 0.8338598987097116 0.5836331144058038 -4.424375178452462 4.267669725713981 7.926703435417678; -2.8110514378395957 -2.829639929972228 -14.770246747583668 -2.559976679926816 -3.2194266644067833; 1.123511526789296 8.988571794122638 17.16247160973714 4.514259718454757 19.2984359834591], bias = [9.537905695874917; 2.049515884531683; 4.248876675671093; 9.099886370118856; -15.882322265944742]), layer_3 = (weight = [17.109555643042683 -2.968767163216418 -4.658757793419889 8.274980631669644 -5.938253824975476; 1.7555736875992336 9.282667521376071 2.07405653174241 1.5360827630811602 1.415347138010818; -2.2166629938399116 -4.817020004083972 3.669632262657891 -1.5434346026328534 -0.6591874714525724; 13.560604531822984 1.5136686735532974 -8.64066548603953 7.2083930982249464 13.650163045234647; 1.0281361434179388 -6.587606208456828 2.5188093421485584 2.659175348735485 -2.914733288425055], bias = [1.4328021361081722; 1.202403085024456; -1.2176474679140588; 0.8122208700141326; 0.05202680107995734]), layer_4 = (weight = [-2.8033417612054103 0.584138431064306 -0.019601293592080224 3.4802840445676 -1.692868112225231; 1.015918998053261 8.696745383573063 -0.220589083797818 -0.4434686844381981 1.3651228855969768], bias = [0.5579418976750097; -1.217529131811827]))

# defining the time span for the plot
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Trained_parameters\\p_minimized_moderatenoise.txt","w") do f

    write(f, string(p_trained))
end
p_trained = (layer_1 = (weight = [5.434005994995207 -2.5992722478801977; -9.57733591004127 -10.634827473142202; -0.9967072322479994 -1.3982282512018958; 7.147376942835671 4.679283095452256; -5.848607833623388 -16.477837822920335], bias = [-6.276257084737525; -1.135550712553739; 0.06728830421322665; 3.1886444680165953; -1.4781405560903216]), layer_2 = (weight = [-4.648768502784665 -0.1110103985478996 -10.822135400886754 -1.3256141561034285 -8.112330068062928; -1.880172158940449 -9.712409688097432 -2.8460938042711867 -8.64097552589408 -1.1708312641391887; 0.8338598987097116 0.5836331144058038 -4.424375178452462 4.267669725713981 7.926703435417678; -2.8110514378395957 -2.829639929972228 -14.770246747583668 -2.559976679926816 -3.2194266644067833; 1.123511526789296 8.988571794122638 17.16247160973714 4.514259718454757 19.2984359834591], bias = [9.537905695874917; 2.049515884531683; 4.248876675671093; 9.099886370118856; -15.882322265944742]), layer_3 = (weight = [17.109555643042683 -2.968767163216418 -4.658757793419889 8.274980631669644 -5.938253824975476; 1.7555736875992336 9.282667521376071 2.07405653174241 1.5360827630811602 1.415347138010818; -2.2166629938399116 -4.817020004083972 3.669632262657891 -1.5434346026328534 -0.6591874714525724; 13.560604531822984 1.5136686735532974 -8.64066548603953 7.2083930982249464 13.650163045234647; 1.0281361434179388 -6.587606208456828 2.5188093421485584 2.659175348735485 -2.914733288425055], bias = [1.4328021361081722; 1.202403085024456; -1.2176474679140588; 0.8122208700141326; 0.05202680107995734]), layer_4 = (weight = [-2.8033417612054103 0.584138431064306 -0.019601293592080224 3.4802840445676 -1.692868112225231; 1.015918998053261 8.696745383573063 -0.220589083797818 -0.4434686844381981 1.3651228855969768], bias = [0.5579418976750097; -1.217529131811827]))
#Retrieving the Data predicted for the White Dwarf model, with the UDE with the trained parameters for the NN
X̂ = predict_ude(p_trained)

# Plot the UDE approximation for  the White Dwarf model
pl_trajectory = scatter(etasteps2, transpose(X̂),markeralpha=0.4, xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth noisy data 
scatter!(etasteps2, transpose(training_array),title="Trained UDE vs Noisy Data", color = :black,markeralpha=0.4, label = ["Noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\UDE_trainedvsData_moderate_noiseI.png")


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

p=p_trained

#UDE prediction
prob_node_extrapolate = ODEProblem(recovered_dynamics!,I, etaspan,p_trained)
_sol_node = solve(prob_node_extrapolate, Tsit5(),abstol=1e-15, reltol=1e-15,saveat = etasteps)

#UDE Extrapolation scatter plot
predicted_ude_plot = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
#UDE trained against training data
pl_trajectory = plot!(etasteps2, transpose(X̂), xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\trainedUDE90points_vsforecasted_udeI.png")




# Producing a scatter plot for the ground truth noisy data 
scatter(etasteps,transpose(x1_noise), color = :blue,markeralpha=0.5, label = ["Ground truth Noisy data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.2,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsNoisy_groundtruth_dataI.png")

# Producing a scatter plot for the ground truth ODE data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.5,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsODE_groundtruth_data.png")


#plot()

#Final Plot
scatter(sol.t[1:end-10],Array(x1_noise[:,1:end-10])[1,:],color=:blue,markershape=:cross, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t[1:end-10],Array(x1_noise[:,1:end-10])[2,:],color=:blue,markershape=:cross, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi NODE
scatter!(collect(etasteps[1:end-10]), X̂[1,:],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps[1:end-10]), X̂[2,:],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-9:end],_sol_node[1,end-9:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-9:end],_sol_node[2, end-9:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\Whitedwarf_forecasted_modelUDE_Finalplot.png")

#Second version
#plot!(_sol_node)

#Last plot 
#Last Version for the preprint

#----------------------------------

scatter(sol.t[1:end-10],Array(x1_noise[:,1:end-10])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-09:end],Array(x1_noise[:,91:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(sol.t[1:end-10],predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-10:end],_sol_node[1,end-10:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\NeuralODEModel_finalversion.png")



# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:90],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-10:end],p_trained,st)[1]

plot!(sol.t[90:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.45, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :right)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0\\UDE\\Results\\ModerateNoise\\Recoveredterm2_nonoise.png")




