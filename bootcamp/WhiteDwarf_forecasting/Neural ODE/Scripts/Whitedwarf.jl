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




#Constants
C = 0.01


#Initial Conditions
I = [1, 0]   #Psi(0)=1, Psi'(0)=1
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

etasteps =  etasteps[6:end-5]
etaspan = (etasteps[1], etasteps[end])
I=[0.984864,-0.10076]
prob_neuralode = NeuralODE(dudt2, etaspan, Tsit5(); saveat = etasteps)

function predict_neuralode(p)
    Array(prob_neuralode(I, p, st)[1])
end
#Training data
true_data= Array(sol[:,6:end-5])
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

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.1); callback = callback,
    maxiters = 80)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false, maxiters=100)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)

xlabel!("Eta (dimensionless radius)")


savefig("C:\\Users\\Raymundoneo\\Documents\\SciML Workshop\\bootcamp\\WhiteDwarf_forecasting\\Neural ODE\\Results\\Whitedwarf_no_noise_ODE.png")


#---------------------Forecasting-----------------------#
#------------------------------------------------------

#------------------------------------------------------


function dudt_node(u,p,t)
    phi, phiderivative = u
   
    output, _ = dudt2([phi,phiderivative],p,st)
    dphi, dphiderivative = output[1],output[2]
    return [dphi,dphiderivative]
end

#Initial Conditions
I = [1, 0]   #Psi(0)=1, Psi'(0)=1
etaspan2 = (0.05, 5.325)

#radius range
datasize= 100
etasteps2 = range(etaspan2[1], etaspan2[2]; length = datasize)



#Neural ODE prediction
prob_node_extrapolate = ODEProblem(dudt_node,I, etaspan2, result_neuralode2.minimizer)
_sol_node = solve(prob_node_extrapolate, Tsit5(),saveat = collect(etasteps2))
p_node = scatter(_sol_node)





I=[0.984864,-0.10076]
p=result_neuralode2.minimizer
predict=Array(prob_neuralode(I, p, st)[1])
prob_neuralode = ODEProblem(dudt2_node, I, etaspan; saveat = etasteps)


pred = Array(prob_neuralode(I, p, st)[1])


scatter!(collect(etasteps), predict[1, :],markershape=:xcross; label = "\\phi prediction")

scatter!(collect(etasteps), predict[2, :],markershape=:xcross; label = "\\phi ' prediction")



