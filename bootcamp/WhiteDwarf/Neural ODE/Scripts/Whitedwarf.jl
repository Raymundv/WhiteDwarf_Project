#Loading the necessary libraries
using Plots
using DifferentialEquations
using Random
using Statistics
using OrdinaryDiffEq
using Lux 
using DiffEqFlux
using ComponentArrays 
using Optimization, OptimizationOptimJL, OptimizationOptimisers                                                                   
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


prob_neuralode = NeuralODE(dudt2, etaspan, Tsit5(); saveat = etasteps)

function predict_neuralode(p)
    Array(prob_neuralode(I, p, st)[1])
end

true_data= Array(sol)
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

        plt1 = scatter(sol.t, true_data[1, :]; label = "\\phi data")
        scatter!(plt1, sol.t, pred[1, :],markershape=:xcross; label = "\\phi prediction")
        scatter!(plt1, sol.t, true_data[2, :]; label = "\\phi ' data")
        scatter!(plt1, sol.t, pred[2, :],markershape=:xcross; label = "\\phi ' prediction")
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

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.1); callback = callback,
    maxiters = 80)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false, maxiters=100)

callback = function (p, l, pred; doplot = true)
    println(l)
        # plot current prediction against data
    if doplot
    
        plt1 = scatter(sol.t, true_data[1, :],color = :blue,title="Trained Neural ODE",markeralpha=0.30; label = "\\phi data")
        scatter!(plt1, sol.t, pred[1, :],markershape = :xcross; label = "\\phi predicted")
        scatter!(plt1, sol.t, true_data[2, :],color = :red,markeralpha=0.4; label = "\\phi' data")
        scatter!(plt1, sol.t, pred[2, :],markershape=:xcross,color= :black; label = "\\phi' predicted")
            #plt1 = scatter(sol.t, true_data[3, :]; label = "data")
            #scatter!(plt1, sol.t, pred[3, :]; label = "prediction")
            #plt=plot(plt1, plt2)
            
        display(plot(plt1))
    
            
            
    end
    return false
end
    
open("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\Neural ODE\\Trained_parameters\\p_minimized_nonoise.txt","w") do f

    write(f, string(result_neuralode2.minimizer))
end

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)
#p=res.minimizer
#callback(p, loss_neuralode(p)...; doplot = true)
#
xlabel!("\\eta (dimensionless radius)")

savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\Neural ODE\\Results\\Whitedwarf_no_noise_ODE.png")

#Final plot for the preprint 
#Last Version for the preprint

#----------------------------------





scatter(sol.t,Array(sol[:,1:end])[1,:],color=:blue,markeralpha=0.3, linewidth = 1, xaxis = "\\eta",
     label = "Training \\phi ", title="White Dwarf model")

scatter!(sol.t,Array(sol[:,1:end])[2,:],color=:blue,markeralpha=0.3, linewidth = 1,markershape=:diamond, xaxis = "\\eta",
     label = "Training \\phi' ", title="Trained Neural ODE")


#scatter!(sol.t[1:end],Array(sol[:,1:end])[1,:], color=:red,markeralpha=0.3, label = "Testing \\phi")

plot!(sol.t[1:end],predict_neuralode(p_trained)[1, :],color=:black,markeralpha=0.3; label = "Predicted \\phi")
xlabel!("\\eta (dimensionless radius)")




plot!(sol.t[end-99:end],predict_neuralode(p_trained)[2, :],color=:black,linestyle=:dash,label="Predicted \\phi'")
title!("Trained Neural ODE")
savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf\\Neural ODE\\Results\\NeuralODEModel_finalversion_nonoise.png")

