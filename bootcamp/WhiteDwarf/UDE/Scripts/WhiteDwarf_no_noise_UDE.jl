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
I = [1, 0]   #Psi(0)=1, Psi'(0)=1
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

solutionarray = Array(sol)
# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,solutionarray[:,1], etaspan, p)

#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)

function predict(theta, X = solutionarray[:,1], T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end


# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict(theta)
    sum(abs2, solutionarray .- X̂)
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
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\losses_no_noise.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer


#p_trained=(layer_1 = (weight = [-1.512504527572715 0.3707509785316587; -2.306297942611551 0.05896917491265696; -2.5672979170900354 0.7828013391501035; -0.705907178826173 -1.8298463543034704; 1.4184752175343749 -0.1536978491203952], bias = [-0.24886240519012826; -1.2746863354814622; -2.563677028347165; -0.7270258749959325; 2.013438320582101]), layer_2 = (weight = [0.31288770590421655 1.3658415973124631 -0.8532336203315337 -1.0440135808452418 0.4053916385620475; 2.0143113626617746 1.4622213730967382 0.8840043088195745 1.8052415767397185 1.003935836329105; 1.6236891217838822 -1.0435208303827492 0.10749924319046081 1.5892288622933906 -0.34418630064992595; 1.751985139220737 1.0575607944603767 0.585691684978437 1.7759037550369328 1.055584710849473; 0.7401258334848936 2.0847635641614644 2.050075926734845 1.5160846556920515 2.0872942090395576], bias = [-0.16238198373375523; 1.6178191140883822; 0.1919388124487184; 1.8705689622623334; 0.3075805108717548]), layer_3 = (weight = [-0.6963956338560692 0.06868519654005839 0.44468847298858755 -0.7852707027725997 1.0436053630477158; 1.727214248182171 1.9344332727964784 1.5173909271935542 1.689205765352056 0.735437263176112; 1.3381248076116883 0.6022837821675663 0.91769886810118 1.334839333350124 0.9210608566907901; 0.9636211126581653 1.403108892288453 0.0674736278132482 0.27482350564250546 1.692997468095379; 1.6408064175001185 1.419267813923653 1.1797947775972821 1.4802068008654723 0.9026308099879142], bias = [-0.6452529207619881; 1.2056577626145777; 0.7002512063199803; 0.5415832913196543; 1.568188623196322]), layer_4 = (weight = [-0.0025086708708675024 -0.1710344428565627 0.07093713591657648 -0.025251161961447673 -0.9838343131263864; -1.0927187475584033 -0.17731716261079236 1.5591354954384333 0.8044151253993317 -1.1735796602205832], bias = [0.002330693101030373; 0.07177955278852739]))
# defining the time span for the plot
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Trained_parameters\\p_minimized_nonoise.txt","w") do f

    write(f, string(res1.minimizer))
end

#Retrieving the Data predicted for the Lotka Volterra model, with the UDE with the trained parameters for the NN
X̂ = predict(p_trained, solutionarray[:,1], etasteps)

# Plot the UDE approximation for  the White Dwarf equation
pl_trajectory = scatter(etasteps, transpose(X̂)[:,1],title="Trained UDE", xlabel = "\\eta (dimensionless radius)", color = :blue, markeralpha=0.30, label = "Training \\phi")
                scatter!(etasteps, transpose(X̂)[:,2],color = :red,markeralpha=0.4,label = "Training \\phi'")
# Producing a scatter plot for the ground truth data 
scatter!(sol.t, solutionarray[1, :],markershape = :xcross,color= :red; label = "\\phi predicted")
scatter!(sol.t, solutionarray[2, :],markershape=:xcross,color= :black; label = "\\phi' predicted")
xlabel!("\\eta (dimensionless radius)")


#scatter!(sol.t, transpose(solutionarray), color = :blue,markeralpha=0.4, label = ["Ground truth data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\UDEvsODE_no_noise_better")

#Iproduced the new plot
#savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\UDEvsODE_no_noise_better_formated")
