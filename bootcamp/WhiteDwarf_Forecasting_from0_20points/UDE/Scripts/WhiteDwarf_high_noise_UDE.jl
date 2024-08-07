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
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 1100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\losses_moderate_noise2I.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer



# defining the time span for the plot


p_trained = (layer_1 = (weight = [-1.8193159452164203 2.744073559928421; 1.8114828090663073 -2.824422799463003; -1.1871640793049103 0.1991819345565677; -1.9797832626046368 -1.274215682796942; -2.9797708415831488 -0.5704117339668496], bias = [-1.9753779669633234; 1.8315393640754296; -1.819746187913108; -1.2626804101770939; -2.157606056595257]), layer_2 = (weight = [-1.656996600323156 -1.0219355838381443 -2.4718419206721456 -1.9329616455713972 -1.590472057068167; 2.2899365632163615 1.5408609121100554 1.3239409821240542 0.8208608706629149 1.9876863575368522; 0.7363991094295163 1.305363764437111 -0.6198462602084744 0.7490730937937635 -0.15184425974692353; -0.012629502432272646 0.8725598564012396 -0.6364575074541449 0.23592389370334638 0.058425763341647126; 0.9121718146070315 1.2519985935044466 1.2542535069996472 0.819289787390472 0.7079388404595489], bias = [-1.623987413618985; 1.5413492780932259; -0.03380050160553218; 0.036630168215904216; 0.6782680019010362]), layer_3 = (weight = [1.9759394170174651 1.5395373068511884 1.7051080679882484 2.1932001329092166 1.195254885897714; -1.59460929730924 -1.0080664788216198 -2.3001334835338882 -2.334234455912098 -2.550182176191753; -2.554137033554578 -1.7721260902612217 -2.0789249193115116 -2.777366494470448 -2.7829394649623036; 1.7240012209562932 0.7872925611487056 1.5973228673486781 1.2533588035800738 1.1714516173926828; -0.9949778922259092 -1.52495466337609 -1.5605743559688732 -1.5867058292853926 -1.4617828914896809], bias = [1.6111057626333802; -2.2011852385946535; -2.235313483555916; 1.303586508751524; -1.4244039633051124]), layer_4 = (weight = [0.2868896085245463 -1.3326704793504212 -0.9387036268390881 0.5881310032855239 -0.4237699378572846; -0.9862025525458352 -1.5063592971174946 -1.1699786455554597 -0.0042641372604034205 -0.772597586415812], bias = [0.1539669615205976; -0.7738596664851816]))
#Retrieving the Data predicted for the Lotka Volterra model, with the UDE with the trained parameters for the NN
p = p_trained
X̂ = predict_ude(p_trained)

# Plot the UDE approximation for  the CWDE model
pl_trajectory = scatter(etasteps2, transpose(X̂),markeralpha=0.4, xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth noisy data 
scatter!(etasteps2, transpose(training_array),title="Trained UDE vs Noisy Data", color = :black,markeralpha=0.4, label = ["Noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\UDE_trainedvsData_moderate_noiseI.png")


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


open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Trained_parameters\\p_minimized_highnoise.txt","w") do f

    write(f, string(res1.minimizer))
end


#UDE prediction
prob_node_extrapolate = ODEProblem(recovered_dynamics!,I, etaspan,p)
_sol_node = solve(prob_node_extrapolate, Tsit5(),abstol=1e-15, reltol=1e-15,saveat = etasteps)

#UDE Extrapolation scatter plot
predicted_ude_plot = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
#UDE trained against training data
pl_trajectory = plot!(etasteps2, transpose(X̂), xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\trainedUDE90points_vsforecasted_udeI.png")




# Producing a scatter plot for the ground truth noisy data 
scatter(etasteps,transpose(x1_noise), color = :blue,markeralpha=0.5, label = ["Ground truth Noisy data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.2,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\UDE_Forecasted_vsNoisy_groundtruth_dataI.png")

# Producing a scatter plot for the ground truth ODE data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=0.5,color=:red, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\UDE_Forecasted_vsODE_groundtruth_dataI.png")


#Final plot for the results- better formated
scatter(sol.t[1:end-80],Array(x1_noise[:,1:end-80])[1,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t[1:end-80],Array(x1_noise[:,1:end-80])[2,:],color=:blue, markershape=:cross, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi NODE
scatter!(collect(etasteps[1:end-80]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps[1:end-80]), predict_ude(p_trained, solutionarray[:,1], etasteps2)[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-79:end],_sol_node[1,end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-79:end],_sol_node[2, end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\Whitedwarf_forecasted_modelUDE.png")

#Second version



#plot!(_sol_node)


# I changed to 800 hundred, due to the jumpings

#Final plot for the preprint 

#Last Version for the preprint


scatter(sol.t[1:end-80],Array(x1_noise[:,1:end-80])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-79:end],Array(x1_noise[:,21:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(sol.t[1:end-80],predict_ude(p_trained, solutionarray[:,1], etasteps2)[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-80:end],_sol_node[1,end-80:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\NeuralODEModel_finalversion.png")

# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:20],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-80:end],p_trained,st)[1]

plot!(sol.t[20:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.25, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :right)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\UDE\\Results\\HighNoise\\Recoveredterm2_nonoise.png")


