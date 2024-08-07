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
etasteps2=etasteps[1:end-80]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-80]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-80]

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
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\losses_moderate_noise2I.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer

p_trained = (layer_1 = (weight = [-1.515046736889739 1.4715772245693186; 1.9507636663125953 -1.81919361098221; 1.9741221996818683 -3.105294964440201; -1.9861456160038335 -1.3341902677042572; -2.3196710788532724 -1.3141409876454335], bias = [-1.7009749646935068; 1.96924157671041; 1.5840826658275742; -1.2892093947163747; -1.62795547478631]), layer_2 = (weight = [-1.435798790881446 -0.8316900125628542 -2.621142900182773 -1.7434279879516832 -1.4676339792139204; 2.0221397314834757 1.2896904786093069 1.1279923279444743 0.6056373331287483 1.8168063091135749; 1.2276024883059222 1.5273511314003574 -0.3394648956520416 0.9287920293454501 0.16159006532292622; -0.23006365041096238 0.805252974672616 -0.6202532290513828 0.19449927962559055 0.054752332478146405; 0.8597793608644825 1.2749792765350365 1.4861472782766112 0.8618851045804071 0.8958020391415344], bias = [-1.2998022689030984; 1.2149796845617464; 0.12663167074773385; 0.11531064684559203; 0.9941531984358769]), layer_3 = (weight = [1.9644601943042648 1.5555751393452593 1.7214386074199188 2.1828031625701114 1.1630394508835613; -1.4749083790428377 -1.2269016088568667 -2.1671288479950443 -2.3791057620358793 -2.819832468474865; -2.8253562552994262 -2.5253283406358484 -2.247845776437411 -2.9588069378969912 -2.9559184852763627; 1.8235031874765784 0.8066004884151378 1.1700424233636308 1.2095351916379218 1.34988273982248; -1.005832355106733 -1.5626995845676028 -1.5886445244325216 -1.595982580866825 -1.4766526613506339], bias = [1.604507356776649; -2.285230568154275; -2.564351269966508; 1.4023711470007345; -1.441890740374194]), layer_4 = (weight = [-0.37588249359197706 -1.4681420713897797 -1.0097507230318084 0.28950046371967686 -0.709842213486935; -1.0084995840033157 -1.5228515612249625 -1.1729265656229195 -0.17645659041616316 -0.7864599917335807], bias = [0.02241619536226371; -0.8038770606255771]))

p = p_trained
# defining the time span for the plot
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Trained_parameters\\p_minimized_moderatenoise.txt","w") do f

    write(f, string(p_trained))
end

#Retrieving the Data predicted for the White Dwarf model, with the UDE with the trained parameters for the NN
X̂ = predict_ude(p_trained)

# Plot the UDE approximation for  the White Dwarf model
pl_trajectory = scatter(etasteps2, transpose(X̂),markeralpha=0.4, xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth noisy data 
scatter!(etasteps2, transpose(training_array),title="Trained UDE vs Noisy Data", color = :black,markeralpha=0.4, label = ["Noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\UDE_trainedvsData_moderate_noiseI.png")


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


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\trainedUDE90points_vsforecasted_udeI.png")




# Producing a scatter plot for the ground truth noisy data 
scatter(etasteps,transpose(x1_noise), color = :blue,markeralpha=0.5, label = ["Ground truth Noisy data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.2,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsNoisy_groundtruth_dataI.png")

# Producing a scatter plot for the ground truth ODE data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.5,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\UDE_Forecasted_vsODE_groundtruth_data.png")


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
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\Whitedwarf_forecasted_modelUDE.png")

#Final version for the preprint 

#Last Version for the preprint


scatter(sol.t[1:end-80],Array(x1_noise[:,1:end-80])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-79:end],Array(x1_noise[:,21:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(sol.t[1:end-80],predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-80:end],_sol_node[1,end-80:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\NeuralODEModel_finalversion.png")


# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:20],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-80:end],p_trained,st)[1]

plot!(sol.t[20:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.35, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :right)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\ModerateNoise\\Recoveredterm2_nonoise.png")


