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

loss((layer_1 = (weight = [-0.3310253804382096 1.4553246942327713; -0.8347061012286425 -0.29015602213285313; -1.224068333611495 -0.205427889974553; 0.24344834923522043 -1.3832105611887975; 0.9490681330138283 1.3138313388793306], bias = [-0.9122039627140001; -0.9610561339454454; -0.5323944694494215; -0.5031909557758517; 0.3040194546753242]), layer_2 = (weight = [0.4920557359424895 0.2439878330787431 -1.0268851249783983 -0.7462301394443603 -0.5112419257459795; 1.2059129719238726 0.5232773342794126 0.15599645455475278 1.0987563583143074 0.07272961731433593; -0.010980205018576612 0.17325469171093186 0.5616475953688546 0.9830245020939153 0.7710571251082626; 0.03767542713067572 -0.22750966291478786 -0.4996096506083295 -0.18127165723038433 -0.24987395660145412; -0.6677171262888651 1.0064408404795424 0.9346387017345634 0.3751385479296065 0.867001436961658], bias = [0.15812058367412157; 0.69803790770096; -0.05692925254421886; -0.11171319832987485; 0.381178378434595]), layer_3 = (weight = [0.039369049249176356 0.44308830323005366 0.7680625273018807 -0.3166756088891275 0.8504337718882388; 1.0761884169587668 1.3301184624377729 0.9195483644541373 1.0369922815829253 0.12151587181032675; 0.6672706577517205 -0.12399362688841908 0.7566318962312022 0.6593455412715742 0.9081360205126349; 0.38036460247965614 1.132779292387273 0.13829329940362606 0.2537121348650556 1.213927480937104; 1.0294327346700105 0.7701335867770361 0.4559064360855069 0.7232847979666785 0.11916172702541047], bias = [-0.042552752995802384; 0.49168865833952047; -0.11891401644650743; 0.5663315683326091; 0.8967878983018832]), layer_4 = (weight = [0.00021591159035941289 0.33184790957471577 0.017571453706137503 -0.014651682170940988 -0.6576649437022624; -0.8729725882115271 -0.14953910934073253 0.9562824417228781 0.9958161155945786 -1.2244740260127254], bias = [0.00023834217964816663; -0.3475714292834833])))


#moderate noise 

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


#------------------Adding moderate noise to data:--------------------#
#--------------------------------------------------------------------#

x1=Array(sol)

x1_mean = mean(x1, dims = 2)
noise_magnitude = 7e-2
x1_noise = x1 .+ (noise_magnitude*x1) .* randn(eltype(x1), size(x1))



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

loss((layer_1 = (weight = [-1.1710321106924275 2.5497610416841736; -1.9752445306473239 -0.4573892100916554; 1.6377064109128705 -1.6081172362584286; -0.7303985644010678 1.2028036295733968; -2.666469851294151 -4.158182508584599], bias = [2.4789480525469876; -0.5203304120726031; -2.58376495214217; -0.19987295628932777; 0.9522062182626337]), layer_2 = (weight = [-0.8223889075285493 -0.245671249126329 -1.5019329964930017 -1.2266193887311756 -0.9767458094840697; 1.5266945600451747 0.8437557491523062 0.3861403515122325 0.19182662415228566 1.4744900266471936; 0.09555174338142916 1.4141114949523585 -0.07886738972050077 -0.01627271170720193 -0.0059473274454871945; -1.8891967044796034 0.17495298568639125 -1.6858974814321348 0.7337557240870672 2.0636936415141816; 1.7107860225266769 -0.28055343033146524 1.824570138282683 -0.5198923476348057 -3.142815750210537], bias = [-0.8620208655134972; 0.8005607677591499; 0.038968763131035415; -0.5409118657528039; 1.6045735416625158]), layer_3 = (weight = [1.3524760726175225 0.8922443764754253 1.3349671999141608 1.550347702983861 0.5288505480719378; -0.34258069672908237 0.320373762561616 -0.5258174328844744 2.061129334557985 -1.7664012555659834; -1.639589106588196 -0.8413368436192847 -0.6579174170861408 -1.8328519239696226 -1.6189572585156096; 1.1910816393941492 0.15273681749379464 -0.3992892980379861 5.035946425021751 2.2925780609874855; -0.4110814092750676 -0.9001970490534819 -1.280353978291379 -3.0580640471003644 0.5174244646128212], bias = [1.6563112746096535; -0.7241227822065467; -1.5171664407406378; 1.1895957543590248; -1.9261569157353373]), layer_4 = (weight = [-0.2260154344518752 -0.5442263109488109 -0.11104108875844558 1.6676848697643798 3.621700523146915; -1.1583621164368048 -1.694296914253846 -0.2777702534383362 -5.970827822431849 -0.8876342761559407], bias = [0.21772962561970657; 0.2181806589011117])))


#high noise 
 

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


#------------------Adding moderate noise to data:--------------------#
#--------------------------------------------------------------------#

x1=Array(sol)

x1_mean = mean(x1, dims = 2)
noise_magnitude = 35e-2
x1_noise = x1 .+ (noise_magnitude*x1) .* randn(eltype(x1), size(x1))



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

loss((layer_1 = (weight = [-1.567316271195498 -0.9216318933955566; -1.787292503457178 -0.6759406386557488; -0.6948343426553105 -0.4606772896221909; -2.2529477868387526 1.2214351198991988; -1.8678628735057883 -1.4683904157625742], bias = [-1.66385435414385; -1.7519604708403422; -1.6155886004415354; -1.6258139480569476; -1.2358590417683757]), layer_2 = (weight = [-1.2577146982545682 -0.6886468417225953 -2.2296825614802764 -1.6241713943766756 -1.3806466954583987; 2.106787908921441 1.4142801633102768 1.364842473555346 0.6464295506728144 1.9500866671569925; 1.211588400496942 1.427461371125084 -0.6322224289263189 0.6888368910955939 -0.14221225599731882; -0.7468674026845075 -0.06873937677288999 -0.6099932758781293 -0.20837492784033665 -0.4873153203685807; 0.4193302292982903 0.7733409309595426 -0.11620908447171212 0.05695189927426096 -0.07058877093245552], bias = [-1.0978714351152061; 1.4955956637014896; 0.19046756060855302; -0.19649180255284787; -0.21063014298132174]), layer_3 = (weight = [1.9238819615537468 1.5081552279569188 1.6857913153272854 2.102489150031562 1.062600073899889; -2.0775802195881328 -1.5616437576833242 -2.5821669237137974 -2.1897870856799075 -2.75393038562553; -2.920347665753938 -2.260786601118151 -1.865365792862753 -2.63266271642411 -2.9225201641932967; 1.839176241369884 0.8564540181954057 1.2384098663731784 1.059893843118218 1.9409821165077408; -0.9889823917105702 -1.5340086200913516 -1.5511629966592586 -1.5641214671299077 -1.4287503333442721], bias = [1.6035740673430354; -2.070902050430901; -2.0360506559737743; 1.2958467610133086; -1.439479423501944]), layer_4 = (weight = [-0.09288956951583764 -1.3926137511093306 -0.8317325507688428 0.4148214831055747 -0.5630208587764481; -0.45510074166063463 -1.3439348241423268 -0.948542004054094 -0.20214259722159883 -0.48427590211991206], bias = [-0.0019758379558647084; -0.548973815129146])))

