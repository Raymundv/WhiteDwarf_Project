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
loss((layer_1 = (weight = [-1.512504527572715 0.3707509785316587; -2.306297942611551 0.05896917491265696; -2.5672979170900354 0.7828013391501035; -0.705907178826173 -1.8298463543034704; 1.4184752175343749 -0.1536978491203952], bias = [-0.24886240519012826; -1.2746863354814622; -2.563677028347165; -0.7270258749959325; 2.013438320582101]), layer_2 = (weight = [0.31288770590421655 1.3658415973124631 -0.8532336203315337 -1.0440135808452418 0.4053916385620475; 2.0143113626617746 1.4622213730967382 0.8840043088195745 1.8052415767397185 1.003935836329105; 1.6236891217838822 -1.0435208303827492 0.10749924319046081 1.5892288622933906 -0.34418630064992595; 1.751985139220737 1.0575607944603767 0.585691684978437 1.7759037550369328 1.055584710849473; 0.7401258334848936 2.0847635641614644 2.050075926734845 1.5160846556920515 2.0872942090395576], bias = [-0.16238198373375523; 1.6178191140883822; 0.1919388124487184; 1.8705689622623334; 0.3075805108717548]), layer_3 = (weight = [-0.6963956338560692 0.06868519654005839 0.44468847298858755 -0.7852707027725997 1.0436053630477158; 1.727214248182171 1.9344332727964784 1.5173909271935542 1.689205765352056 0.735437263176112; 1.3381248076116883 0.6022837821675663 0.91769886810118 1.334839333350124 0.9210608566907901; 0.9636211126581653 1.403108892288453 0.0674736278132482 0.27482350564250546 1.692997468095379; 1.6408064175001185 1.419267813923653 1.1797947775972821 1.4802068008654723 0.9026308099879142], bias = [-0.6452529207619881; 1.2056577626145777; 0.7002512063199803; 0.5415832913196543; 1.568188623196322]), layer_4 = (weight = [-0.0025086708708675024 -0.1710344428565627 0.07093713591657648 -0.025251161961447673 -0.9838343131263864; -1.0927187475584033 -0.17731716261079236 1.5591354954384333 0.8044151253993317 -1.1735796602205832], bias = [0.002330693101030373; 0.07177955278852739])))

#moderate noise 



rng = Random.default_rng()
Random.seed!(99)
#Adding moderate noise to data:

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

p_trained_moderate = (layer_1 = (weight = [-0.34491090516130724 0.12462421069487915; -1.0161608650980276 -1.4750952540061166; -0.30712570849905296 -0.20945073569626335; 0.66618932565025 -0.020881231600814554; 0.13880758428305331 -1.46219504266019], bias = [0.18855672332646017; -0.6618432041111132; -0.28521881315803943; 0.4741759572016006; 1.0357974890959363]), layer_2 = (weight = [-0.8145186916750358 -0.22993998715635353 -1.363211285654756 -1.0946184781647998 -0.8659563870280454; 1.5936248608862582 0.8864133420036717 0.4639529376366866 0.2325256552310734 1.486816206018957; 0.35328927436389257 1.350106641921453 -0.0787713773433934 0.1215643000666599 -0.8442323107055106; -1.106142684890842 -0.1670836242777411 -1.05590630159697 -0.6829791721576194 -0.7006200623633843; 2.8364798654717673 1.5210441801664019 3.4882516384877658 2.4214822796658004 1.4560338599193732], bias = [-0.7814265038305162; 0.8348796519189116; -0.04833396129840622; -0.8317753311019869; 3.2252993545329014]), layer_3 = (weight = [1.2349723735222151 0.7982000644795328 2.149340343074601 1.12637596592888 0.17086457615126527; 0.03327103823341647 0.5131369545701155 1.055202125021544 -0.5095798101770289 -1.0325869636123388; -1.6838484883668114 -0.8531530232599136 0.026166671720465377 -1.5724836799970603 -1.6169485408855035; 1.1863121898655222 0.2794231189183038 -0.1308954788124877 0.49915824072092896 0.8330590116078567; -0.3294029729077209 -0.8619273949389147 -1.2096668666246484 -0.6888777938567552 -0.5476862455812513], bias = [2.459620963063957; 0.4360696266515581; -1.3607231822176706; 0.47259202939130796; -0.11070783332441746]), layer_4 = (weight = [1.0757542526704578 -0.26025706222249556 0.3836190671749298 0.2580083547883666 0.190264425363767; 1.3800351546993483 0.4647973444923918 0.3430570821943228 -0.462839820616631 1.5223548117820973], bias = [-0.2910728128192493; -0.9378313489435097]))
loss(p_trained_moderate)


#------------------------high noise ---------------------#
#--------------------------------------------------------#
rng = Random.default_rng()
Random.seed!(99)
#Adding moderate noise to data:

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

p_trained_moderate = (layer_1 = (weight = [-0.8901191381367268 -4.712719172855476; -1.395617491250587 -1.804574210577113; -0.7344898093032929 -1.942212612334039; -0.44711712479711585 1.023016097056906; -6.625656609687567 -6.684580293067858], bias = [-0.10833792597901784; -1.769100133641662; -1.1923323305783946; -1.9091544455156515; 1.2625439221333146]), layer_2 = (weight = [-1.5793628368178416 -0.8415109049193342 -2.3706479701641006 -1.9657528990633941 -1.7420865925877995; 2.153021509563261 0.2847678614209927 2.5421532820478694 -0.11720926929394745 8.059755757015022; 1.8300323736445845 1.9221380055463675 1.3961313052140738 1.5241758902254563 0.8989351178156105; -2.029645018736675 -1.0120032518660862 -1.6057072346046715 -1.4042035900852774 -1.2243428990077976; 1.3372884198747854 1.6754062589766479 1.521857091120925 1.100357924551906 1.2934847603823685], bias = [-1.9242574764125826; -1.707623590699739; 1.8129137779089604; -1.8227284775018369; 2.0260013649889337]), layer_3 = (weight = [1.821843677673886 10.85960612013984 1.5819958735703457 1.7055252065097175 0.5479163886500312; 1.6794059250095674 2.2075552500439084 1.426583728931246 1.3915263313295299 0.7973136477407702; -2.372429818236951 -0.9366538781345051 -1.1009851383555125 -2.190365554650455 -2.014984043703852; 1.725216737603289 -8.5400209634787 0.4633640732735316 1.0450346581383982 1.1765699671533703; -0.9033007189119121 0.517238968630953 -1.470613898684292 -1.2788883397688258 -1.0306158273718942], bias = [0.01973765031904054; 4.620851709653557; -2.021850740118817; 4.0415154372370194; -0.15343519801237002]), layer_4 = (weight = [-0.8349917219432856 -0.4320754065343898 -0.5881330874267477 -6.037880504055824 -7.757974329588875; 1.0479067963597841 -0.5490514005973874 0.37853753268092516 2.547059291338586 7.99070463241802], bias = [8.379970532777392; -8.882872523303774]))
loss(p_trained_moderate)