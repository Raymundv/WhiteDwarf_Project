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
etasteps2=etasteps[1:end-90]
etaspan2 = (etasteps2[1],etasteps2[end])
prob_NN = ODEProblem(ude_dynamics,solutionarray[:,1], etaspan2, p)

#-------------------------Implementing the training routines-------------------------
eta=sol.t[1:end-90]


## Function to train the network (the predictor)

function predictude(theta, X = solutionarray[:,1], T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

training_array=solutionarray[:,1:end-90]
# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predictude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [2.2713597865557293 0.006244847568006231; 1.8048517542423121 -3.3469602997349206; -2.121961760711527 1.4202925014970595; -1.2107412800720905 0.92094783115506; 1.7323904505331973 -0.7541666338100455], bias = [-3.213909494099596; 2.2851815791472436; -1.2990039639961568; -1.928141226913103; 1.4125947767710043]), layer_2 = (weight = [-0.250247648045413 0.25535433430825555 -0.393268158351193 -1.2120900838239155 -0.2539870134368169; 1.8724521821029083 1.3715815459819998 0.791216948510186 1.978301457383517 0.8234973560863208; -0.1241083255546462 -1.1566967406070627 -1.209623224452429 -0.2965037802822515 -0.7000024963572081; -1.426602631725679 -0.7494664143569162 -0.14205739721771704 -0.728464026482059 -0.13333948503573473; -0.08665033996382697 1.5361688855664806 1.584685388383774 1.1291799193976428 1.4410515748948938], bias = [0.0472889687293012; 4.751048695324952; -0.7034397742906747; -0.810668988173282; 1.7578242604075858]), layer_3 = (weight = [-1.0313576741547323 -0.6825424402211429 0.2634075131839641 -1.436889887650714 0.7045107765948037; 1.6856169500750684 1.9356774689847245 1.5217336571852438 1.655601115602615 0.7308540711899718; 0.5489632884611606 -0.4265105618540637 0.5078239539924188 2.389850442602374 -2.0792845583674415; 1.1420574076524699 1.6983846772492166 0.6587655508459727 0.23143697139039296 1.8803888457259996; 1.8936935146128908 1.6599012921273362 1.2424645019012683 1.5890411265192377 1.3874420380773016], bias = [-0.7438842209037831; 1.1874473357014634; -0.012670005440637311; 0.9880276524375143; 1.5979392592266533]), layer_4 = (weight = [1.5340256078620307 -0.4301433312902439 -0.17822426582668935 -0.10605285397343372 -1.2071520644399312; -4.127020994548727 -0.7544678617534741 -1.1499588836438779 -1.040267306662434 -2.1101149518127267], bias = [-0.00333332129020867; -0.43824514399337144])))

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
etasteps2=etasteps[1:end-90]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)


#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-90]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-90]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [5.077156451573874 7.137186877510023; 1.8018128269842208 -0.5975681071842939; 0.8519706774409836 1.7512322234756346; -1.8004716402255796 -1.6809355228465797; -3.2541323424284236 -1.9648136129151026], bias = [0.030246795219938748; 1.8045822813676136; -1.3456298625998744; -1.051214065363055; -2.3807676163214686]), layer_2 = (weight = [-1.8092651197156537 -1.1063215707124705 -2.4751831476204886 -1.8936194269134472 -1.7430736145632457; 0.8617384799842778 1.2314361001476135 2.8345043589008774 0.5825637300584665 1.7672063496060937; 15.558547315822146 1.4870697397937098 8.638195792232912 0.9988500617951418 0.3184167394705994; 51.33275307951912 0.8265486152136737 -67.44646562278776 0.38208144754723117 -0.9281517012848063; 2.7551928642617627 1.2255328799034124 1.890079679410556 0.8037435463799448 0.8825754466086163], bias = [-2.186189967374538; -0.48733228245635724; 14.696334693057645; 45.733165201096284; 1.7207807312411143]), layer_3 = (weight = [2.13531389282426 1.5602158074828065 1.7114237003063804 2.459595874344665 1.2656095797858768; -1.7197287811009279 -1.9867810745242869 -1.7736394042897168 -3.069762507121461 -2.6152861690527582; -2.7287590638747847 -1.8518809688531948 -1.5709499096256583 -2.8599403500460294 -2.6062525127432004; 1.7093486660367645 3.2145116415728614 -3.7611551883330314 7.675961220033988 1.2545356000769583; -1.1123201820843633 -1.7240627190221187 -1.2390286487571278 -3.3297546105548204 -1.5793349497634346], bias = [4.552018972616008; -1.850688604908136; -0.43801099107958275; 1.151037288385348; -5.811776472373979]), layer_4 = (weight = [-0.33966086701221654 -2.20983308674166 -2.213073103076914 -34.76811413064417 -2.311776195931363; -0.34638670418974077 -1.35029428292262 -0.2087780745420773 14.032812110790971 1.396733084684465], bias = [0.8715646877432137; -1.0012920777755])))
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
etasteps2=etasteps[1:end-90]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)


#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-90]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-90]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end


