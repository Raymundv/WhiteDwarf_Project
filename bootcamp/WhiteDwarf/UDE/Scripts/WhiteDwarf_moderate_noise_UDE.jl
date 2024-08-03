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


#Adding moderate noise to data:

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
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\losses_moderate_noise.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer

#p_trained = (layer_1 = (weight = [-0.727893842030321 0.13718895950316035; -2.6392009187557726 -1.047837279360003; -2.9767324365373056 -4.338257593448761; -0.4947748264989609 -0.4190833214193289; -2.429146863856118 -2.6985999512748675], bias = [-0.946731243747013; -2.291023618688338; 1.5024091879218837; -0.7425884469046468; 0.1808207728658995]), layer_2 = (weight = [-1.486054159990248 -0.8841407935796622 -3.3825210651832576 -2.0443829941366642 -1.906514949419884; -0.09163625404597203 -0.8691386502607134 -0.733684460675513 -0.5222152846746821 -1.404615824875155; 1.691866525053132 1.9157281173612482 1.9254491090344483 1.5085474809481485 0.6435184622659548; -2.0017585641958546 -1.0252344059876697 -1.828357334575395 -1.4107799646391646 -1.2048112743811372; 1.2711568308379073 1.6959119329418428 2.439661537757485 1.127813572508265 1.2394641382536775], bias = [2.1927081192085414; 0.012453707001215763; 1.5106146606937287; -1.3422224913406786; 2.368620794421298]), layer_3 = (weight = [1.7323931031409627 2.9212579490832624 2.7468942998526664 1.8311887852027546 0.7150504709832053; 1.672543905479793 0.24478999322930586 1.513386828624444 1.3940710938315009 0.7995225076919362; -2.37062346329752 -1.7587401749598668 -1.0991076143624576 -2.191340817246448 -2.0184013380253054; 2.9316359534847645 -0.6548528113140013 -1.2210032243176803 1.119256372358662 1.224769444814523; -0.7688564158152941 -4.124608888676817 -0.8310785157972209 -1.1714237973376354 -0.9305793164526106], bias = [0.8948778213240715; 3.4825280107548062; -1.8698112079447373; 1.1242843738218182; -0.7517543307533492]), layer_4 = (weight = [1.8520325018141501 -0.5430596862639577 -0.07470148717229236 -0.4968973661792778 -1.289380411439508; -0.05253881604643106 0.156937517343783 0.8782616991273586 0.4521136217355267 1.936620717118169], bias = [0.06376589094710307; -0.8960839995622831]))
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Trained_parameters\\p_minimized_moderatenoise.txt","w") do f

    write(f, string(res1.minimizer))
end
# defining the time span for the plot




#Retrieving the Data predicted for the Lotka Volterra model, with the UDE with the trained parameters for the NN
X̂ = predict(p_trained, I, etasteps)

# Plot the UDE approximation for  the a model
#pl_trajectory = plot(etasteps, transpose(X̂),title="Trained UDE", xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])

# Plot the UDE approximation for  the White Dwarf equation
pl_trajectory = scatter(etasteps, transpose(X̂)[:,1],title="Trained UDE", xlabel = "\\eta (dimensionless radius)", label = "\\phi data")
                scatter!(etasteps, transpose(X̂)[:,2],color = :red,markeralpha=0.4,label = "\\phi' data")
# Producing a scatter plot for the ground truth data 
scatter!(sol.t, solutionarray[1, :],markershape = :xcross,color= :red; label = "\\phi predicted")
scatter!(sol.t, solutionarray[2, :],markershape=:xcross,color= :black; label = "\\phi' predicted")
xlabel!("\\eta (dimensionless radius)")

# Producing a scatter plot for the ground truth data 

scatter!(sol.t, transpose(x1_noise), color = :black,markeralpha=0.4, label = ["Ground truth noisy data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\UDEvsODE_moderate_noise")





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
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\UDE\\Results\\NeuralODEModel_finalversion_moderatenoise.png")

