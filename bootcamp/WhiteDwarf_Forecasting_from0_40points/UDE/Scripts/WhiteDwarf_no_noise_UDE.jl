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
etasteps2=etasteps[1:end-60]
etaspan2 = (etasteps2[1],etasteps2[end])
prob_NN = ODEProblem(ude_dynamics,solutionarray[:,1], etaspan2, p)

#-------------------------Implementing the training routines-------------------------
eta=sol.t[1:end-60]


## Function to train the network (the predictor)

function predictude(theta, X = solutionarray[:,1], T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

training_array=solutionarray[:,1:end-60]
# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predictude(theta)
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
res1 = Optimization.solve(optprob1, Optim.BFGS(initial_stepnorm=0.01), callback=callback, maxiters = 1000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")


# Plot the losses for the ADAM routine
pl_losses = plot(1:300, losses[1:300], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
#Plot the losses for the BFGS routine
plot!(301:length(losses), losses[301:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\losses_no_noise.png")
# Retrieving the best candidate after the BFGS training.
p_trained = res1.minimizer


#Saving p_trained for future usage:

open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Trained_parameters\\p_minimized_nonoise.txt","w") do f

    write(f, string(res1.minimizer))
end
# defining the time span for the plot
#implementation of saved p_trained
#p_trained= (layer_1 = (weight = [-0.5914239534866671 -0.12770954089560196; -1.6070369072917845 -0.44245515762800564; -2.4712942739551975 1.1118877430465188; -0.7616332347538572 -1.2060673538996178; 1.318559745587085 -0.14990379393890124], bias = [-0.6334809883945558; -1.0088972605924476; -1.983207841187539; -0.43961441554230724; 1.1666684331014607]), layer_2 = (weight = [0.09562745177399481 0.5174448392528295 -0.8760761051043906 -1.1425452818059387 -0.2464861975181729; 2.043243679586581 1.4536359995135788 0.9094087802399894 1.833006335607427 0.995677046387276; 1.1235385845592747 -0.6805807729047443 -0.3850544478759506 1.3735081262162343 -0.18703819207601377; 1.7131465199152054 1.0582879989376486 0.5980178339750114 1.740286457907272 1.0510009526590338; 0.8825404444455105 1.9440473280508488 1.8895856012975267 1.690059017369644 1.7304100201658927], bias = [0.004656292846577479; 1.6205467951584012; -0.24053553244243153; 1.7899702781324425; 0.2736590585332426]), layer_3 = (weight = [-0.6742941047861514 0.10724929193429811 0.5362797196901873 -0.781577000757492 1.2148775831875176; 1.7105404061977547 1.935338034334163 1.5274962568011754 1.688008504766943 0.7347188785577593; 1.1332514941991276 0.5612931181246403 1.0418858793772148 1.31945010524704 0.75618876657712; 0.8211536503184711 1.4335990009057107 0.2889710409353576 0.2962492034171797 1.6716248706128445; 1.6741626159628524 1.4343704183334411 1.1924310383920635 1.4947586946894378 1.0531370265816813], bias = [-0.4596572201044409; 1.1957717558681744; 0.5168382920997338; 0.5147669273584408; 1.5582827135169408]), layer_4 = (weight = [-0.006147408209046018 -0.2455734080972964 0.03178579014964978 -0.017901843008801974 -1.059489776713348; -1.1324533748217742 -0.31757158914124306 1.443676638574256 0.6562009168047263 -1.2948698293096494], bias = [0.006117786549485974; 0.13476322299608134]))
#Retrieving the Data predicted for the WhiteDwarf Volterra model, with the UDE with the trained parameters for the NN
#p_trained = (layer_1 = (weight = [-0.3310253804382096 1.4553246942327713; -0.8347061012286425 -0.29015602213285313; -1.224068333611495 -0.205427889974553; 0.24344834923522043 -1.3832105611887975; 0.9490681330138283 1.3138313388793306], bias = [-0.9122039627140001; -0.9610561339454454; -0.5323944694494215; -0.5031909557758517; 0.3040194546753242]), layer_2 = (weight = [0.4920557359424895 0.2439878330787431 -1.0268851249783983 -0.7462301394443603 -0.5112419257459795; 1.2059129719238726 0.5232773342794126 0.15599645455475278 1.0987563583143074 0.07272961731433593; -0.010980205018576612 0.17325469171093186 0.5616475953688546 0.9830245020939153 0.7710571251082626; 0.03767542713067572 -0.22750966291478786 -0.4996096506083295 -0.18127165723038433 -0.24987395660145412; -0.6677171262888651 1.0064408404795424 0.9346387017345634 0.3751385479296065 0.867001436961658], bias = [0.15812058367412157; 0.69803790770096; -0.05692925254421886; -0.11171319832987485; 0.381178378434595]), layer_3 = (weight = [0.039369049249176356 0.44308830323005366 0.7680625273018807 -0.3166756088891275 0.8504337718882388; 1.0761884169587668 1.3301184624377729 0.9195483644541373 1.0369922815829253 0.12151587181032675; 0.6672706577517205 -0.12399362688841908 0.7566318962312022 0.6593455412715742 0.9081360205126349; 0.38036460247965614 1.132779292387273 0.13829329940362606 0.2537121348650556 1.213927480937104; 1.0294327346700105 0.7701335867770361 0.4559064360855069 0.7232847979666785 0.11916172702541047], bias = [-0.042552752995802384; 0.49168865833952047; -0.11891401644650743; 0.5663315683326091; 0.8967878983018832]), layer_4 = (weight = [0.00021591159035941289 0.33184790957471577 0.017571453706137503 -0.014651682170940988 -0.6576649437022624; -0.8729725882115271 -0.14953910934073253 0.9562824417228781 0.9958161155945786 -1.2244740260127254], bias = [0.00023834217964816663; -0.3475714292834833]))
X̂ = predictude(p_trained, solutionarray[:,1], etasteps2)

# Plot the UDE approximation for  the WhiteDwarf model
pl_trajectory = plot(etasteps2, transpose(X̂), xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])
# Producing a scatter plot for the ground truth data 
scatter!(etasteps2, transpose(training_array), color = :blue,markeralpha=0.4, label = ["Training data" nothing])
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\trainedUDEvsODE20points.png")

p = p_trained
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


etaspan = (0.05, 5.325)

#radius range
datasize= 100
etasteps = range(etaspan[1], etaspan[end]; length = datasize)



#UDE prediction forecasted
prob_node_extrapolate = ODEProblem(recovered_dynamics!,I, etaspan)
_sol_node = solve(prob_node_extrapolate, Tsit5(),saveat = etasteps)

#UDE Extrapolation scatter plot
predicted_ude_plot = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
#UDE trained against training data
pl_trajectory = plot!(etasteps2, transpose(X̂), xlabel = "\\eta (dimensionless radius)", color = :red, label = ["UDE Approximation" nothing])


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\trainedUDE90points_vsforecasted_ude.png")




# Producing a scatter plot for the ground truth data 
scatter(sol, color = :blue,markeralpha=0.3, label = ["Ground truth ODE data" nothing])
scatter!(_sol_node, legend = :topright,markeralpha=1,markershape=:hline,color=:black, label=["UDE \\phi" "UDE \\phi'"], title="UDE Extrapolation")
xlabel!("\\eta (dimensionless radius)")
#saving 4th figure
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\UDE_Forecasted_vsODE_groundtruth_data.png")



#Final plot for the results- better formated
plot(sol.t[1:end-60],Array(sol[:,1:end-60])[1,:],color=:blue, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

plot!(sol.t[1:end-60],Array(sol[:,1:end-60])[2,:],color=:blue, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi UDE
scatter!(collect(etasteps[1:end-60]), X̂[1,:],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps[1:end-60]), predictude(p_trained, solutionarray[:,1], etasteps2)[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-59:end],_sol_node[1,end-59:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-59:end],_sol_node[2, end-59:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\Whitedwarf_forecasted_modelUDE.png")

#Final actual version for preprint. 

scatter(sol.t[1:end-60],Array(sol[:,1:end-60])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-59:end],Array(sol[:,41:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(etasteps[1:end-60],predictude(p_trained, solutionarray[:,1], etasteps2)[1,:] ,color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-60:end],_sol_node[1,end-60:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained UDE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\NeuralODEModel_finalversion.png")



# Recovering the Guessed term by the UDE for the missing term in the CWDE
Y_guessed = U(X̂,p_trained,st)[1]

plot(sol.t[1:40],Y_guessed[2,:], label = "UDE Approximation", color =:black)


Y_forecasted = U(_sol_node[:, end-60:end],p_trained,st)[1]

plot!(sol.t[40:100], Y_forecasted[2,:], color = :cyan, label = "UDE Forecasted")

function Y_term(psi, C)
    return -((psi^2 - C)^(3/2))
end



Y_actual = [Y_term(psi, C) for psi in Array(sol[:,1:end])[1,:]]

scatter!(sol.t, Y_actual,markeralpha=0.30, color =:orange,label = "Actual term: " * L"-\left(\varphi^2 - C\right)^{3/2}", legend = :right)


title!("UDE missing term")
xlabel!("\\eta (dimensionless radius)")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_40points\\UDE\\Results\\NoNoise\\Recoveredterm2_nonoise.png")




