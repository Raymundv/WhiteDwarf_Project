#Loading the necessary libraries
using Plots
using ModelingToolkit
using DifferentialEquations
using Random
using Statistics
using OrdinaryDiffEq
using Lux 
using DiffEqFlux
using Flux
using ComponentArrays 
using Optimization, OptimizationOptimJL, OptimizationOptimisers                                                                   
rng = Random.default_rng()
Random.seed!(99)

#No Noise forecasting from eta = 0 to the 90th element of the domain etaspan = (0.05, 5.325)


#Constants
C = 0.01


#Initial Conditions
I = [1.0, 0.0]   #Psi(0)=1, Psi'(0)=1
etaspan = (0.05, 5.325)

#radius range
datasize= 100
etasteps = range(etaspan[1], etaspan[2]; length = datasize)


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


#-------------------------------------Defining the Neural ODE------------------------------------


dudt2 = Lux.Chain(Lux.Dense(2, 80, tanh),Lux.Dense(80, 80, tanh), Lux.Dense(80, 2))
#Setting up the NN parameters randomly using the rng instance
p, st = Lux.setup(rng, dudt2)

etasteps =  etasteps[1:end-80]
etaspan = (etasteps[1], etasteps[end])
I=[1.0,0.0]
prob_neuralode = NeuralODE(dudt2, etaspan, Tsit5(); saveat = etasteps)

function predict_neuralode(p)
    Array(prob_neuralode(I, p, st)[1])
end
#Training data
true_data= Array(sol[:,1:end-80])
### Define loss function as the difference between actual ground truth data and Neural ODE prediction
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, true_data .- pred)
    return loss, pred
end


callback = function (p, l, pred; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot

        plt1 = scatter(collect(etasteps), true_data[1, :]; label = "\\phi data")
        scatter!(plt1, collect(etasteps), pred[1, :],markershape=:xcross; label = "\\phi prediction")
        scatter!(plt1, collect(etasteps), true_data[2, :]; label = "\\phi ' data")
        scatter!(plt1, collect(etasteps), pred[2, :],markershape=:xcross; label = "\\phi ' prediction")
        #plt1 = scatter(sol.t, true_data[3, :]; label = "data")
        #scatter!(plt1, sol.t, pred[3, :]; label = "prediction")
        #plt=plot(plt1, plt2)
        
        display(plot(plt1))

        
        
    end
    return false
end


pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot = true)




# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.02); callback = callback,
    maxiters = 150)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.005);
    callback, allow_f_increases = false, maxiters=125)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)

xlabel!("Eta (dimensionless radius)")
title!("Trained Neural ODE vs Ground truth data")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Results\\NoNoise\\Whitedwarf_TrainedNeuralODEvstrainingdata.png")



#Saving the trained parameters
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Trained_parameters\\p_minimized_nonoise.txt","w") do f

    write(f, string(result_neuralode2.minimizer))
end
#---------------------Forecasting-----------------------#
#------------------------------------------------------

#------------------------------------------------------
p_trained = result_neuralode2.minimizer

function dudt_node(u,p,t)
    phi, phiderivative = u
   
    output, _ = dudt2([phi,phiderivative],p,st)
    dphi, dphiderivative = output[1],output[2]
    return [dphi,dphiderivative]
end

#Initial Conditions
I = [1.0, 0.0]   #Psi(0)=1, Psi'(0)=1
etaspan2 = (0.05, 5.325)

#radius range
datasize= 100
etasteps2 = range(etaspan2[1], etaspan2[2]; length = datasize)



#Neural ODE prediction implementation
prob_node_extrapolate = ODEProblem(dudt_node,I, etaspan2, p_trained)
_sol_node = solve(prob_node_extrapolate,saveat = collect(etasteps2))
#Neural ODE Extrapolation scatter plot
p_neuralode = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["NeuralODE \\phi" "NeuralODE \\phi'"], title="Neural ODE Extrapolation")

#Trained (predicted) DATA up to the 90 elements with the Neural ODE, retrieving. Same as callback.
p=result_neuralode2.minimizer
p=p_trained
prob_neuralode = NeuralODE(dudt2,etaspan; saveat = etasteps)
predicted=(prob_neuralode(I, p, st)[1])

#Plot
scatter!(collect(etasteps), predicted[1, :],color=:black,markershape=:hline; label = "\\phi prediction")
xlabel!("\\eta")
scatter!(collect(etasteps), predicted[2, :],color=:black,markershape=:cross; label = "\\phi ' prediction")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Results\\NoNoise\\Whitedwarf_NODEpredictionvspredictedtrainingdata.png")



#Ground truth full data vs Neural ODE full prediction

p_neuralode = scatter(_sol_node, legend = :topright,markeralpha=0.5, label=["NeuralODE \\phi" "NeuralODE \\phi'"], title="Neural ODE Extrapolation")

plot!(sol, linewidth = 1, xaxis = "\\eta",
     label = ["\\phi ODE" "\\phi' ODE"])



savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Results\\NoNoise\\Whitedwarf_no_noise_NODEvsGroundTruthData_ODE.png")

#Final plot for the results- better formated
plot(sol.t[1:end-80],Array(sol[:,1:end-80])[1,:],color=:blue, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

plot!(sol.t[1:end-80],Array(sol[:,1:end-80])[2,:],color=:blue, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi NODE
scatter!(collect(etasteps), predicted[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps), predicted[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-79:end],_sol_node[1,end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-79:end],_sol_node[2, end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")
title!("Trained Neural ODE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Results\\NoNoise\\Whitedwarf_forecasted_model.png")

#Second version

plot(sol.t[1:end-80],Array(sol[:,1:end-80])[1,:],color=:blue, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

plot!(sol.t[1:end-80],Array(sol[:,1:end-80])[2,:],color=:blue, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi'")
xlabel!("\\eta (dimensionless radius)")

#Trained Phi NODE
scatter!(collect(etasteps), predicted[1, :],color=:blue,markeralpha=0.3; label = "Predicted \\phi")

scatter!(collect(etasteps), predicted[2, :],color=:blue, markeralpha=0.3;label = "Predicted \\phi'")
scatter!(sol.t[end-79:end],_sol_node[1,end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi")

scatter!(sol.t[end-79:end],_sol_node[2, end-79:end],color=:orange,markeralpha=0.6,label="Forecasted \\phi'")

scatter!(sol.t[end-79:end],Array(sol)[1,end-79:end],color=:black, markershape=:hline,label="ODE \\phi")
title!("Trained Neural ODE")
scatter!(sol.t[end-79:end],Array(sol)[2,end-79:end],color=:black, markershape=:cross,label="ODE \\phi'")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Results\\NoNoise\\Whitedwarf_forecasted_model_ODEVersion.png")


# Final actual Plot for the preprint
scatter(sol.t[1:end-80],Array(sol[:,1:end-80])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")


scatter!(sol.t[end-79:end],Array(sol[:,21:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(sol.t[1:end-80],predicted[1, :] ,color=:blue,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")

plot!(sol.t[end-80:end],_sol_node[1,end-80:end],color=:red,markeralpha=0.30,label="Forecasted \\phi")
title!("Trained Neural ODE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_Forecasting_from0_20points\\Neural ODE\\Results\\NoNoise\\NeuralODEModel_finalversion.png")





