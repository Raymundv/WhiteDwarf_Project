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
etasteps2=etasteps[1:end-10]
etaspan2 = (etasteps2[1],etasteps2[end])
prob_NN = ODEProblem(ude_dynamics,solutionarray[:,1], etaspan2, p)

#-------------------------Implementing the training routines-------------------------
eta=sol.t[1:end-10]


## Function to train the network (the predictor)

function predictude(theta, X = solutionarray[:,1], T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

training_array=solutionarray[:,1:end-10]
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

loss((layer_1 = (weight = [-0.5914239534866671 -0.12770954089560196; -1.6070369072917845 -0.44245515762800564; -2.4712942739551975 1.1118877430465188; -0.7616332347538572 -1.2060673538996178; 1.318559745587085 -0.14990379393890124], bias = [-0.6334809883945558; -1.0088972605924476; -1.983207841187539; -0.43961441554230724; 1.1666684331014607]), layer_2 = (weight = [0.09562745177399481 0.5174448392528295 -0.8760761051043906 -1.1425452818059387 -0.2464861975181729; 2.043243679586581 1.4536359995135788 0.9094087802399894 1.833006335607427 0.995677046387276; 1.1235385845592747 -0.6805807729047443 -0.3850544478759506 1.3735081262162343 -0.18703819207601377; 1.7131465199152054 1.0582879989376486 0.5980178339750114 1.740286457907272 1.0510009526590338; 0.8825404444455105 1.9440473280508488 1.8895856012975267 1.690059017369644 1.7304100201658927], bias = [0.004656292846577479; 1.6205467951584012; -0.24053553244243153; 1.7899702781324425; 0.2736590585332426]), layer_3 = (weight = [-0.6742941047861514 0.10724929193429811 0.5362797196901873 -0.781577000757492 1.2148775831875176; 1.7105404061977547 1.935338034334163 1.5274962568011754 1.688008504766943 0.7347188785577593; 1.1332514941991276 0.5612931181246403 1.0418858793772148 1.31945010524704 0.75618876657712; 0.8211536503184711 1.4335990009057107 0.2889710409353576 0.2962492034171797 1.6716248706128445; 1.6741626159628524 1.4343704183334411 1.1924310383920635 1.4947586946894378 1.0531370265816813], bias = [-0.4596572201044409; 1.1957717558681744; 0.5168382920997338; 0.5147669273584408; 1.5582827135169408]), layer_4 = (weight = [-0.006147408209046018 -0.2455734080972964 0.03178579014964978 -0.017901843008801974 -1.059489776713348; -1.1324533748217742 -0.31757158914124306 1.443676638574256 0.6562009168047263 -1.2948698293096494], bias = [0.006117786549485974; 0.13476322299608134])))


#moerate noise loss collection 
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
#Displaying true data vs noisy data



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
etasteps2=etasteps[1:end-10]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-10]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-10]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end


loss((layer_1 = (weight = [5.434005994995207 -2.5992722478801977; -9.57733591004127 -10.634827473142202; -0.9967072322479994 -1.3982282512018958; 7.147376942835671 4.679283095452256; -5.848607833623388 -16.477837822920335], bias = [-6.276257084737525; -1.135550712553739; 0.06728830421322665; 3.1886444680165953; -1.4781405560903216]), layer_2 = (weight = [-4.648768502784665 -0.1110103985478996 -10.822135400886754 -1.3256141561034285 -8.112330068062928; -1.880172158940449 -9.712409688097432 -2.8460938042711867 -8.64097552589408 -1.1708312641391887; 0.8338598987097116 0.5836331144058038 -4.424375178452462 4.267669725713981 7.926703435417678; -2.8110514378395957 -2.829639929972228 -14.770246747583668 -2.559976679926816 -3.2194266644067833; 1.123511526789296 8.988571794122638 17.16247160973714 4.514259718454757 19.2984359834591], bias = [9.537905695874917; 2.049515884531683; 4.248876675671093; 9.099886370118856; -15.882322265944742]), layer_3 = (weight = [17.109555643042683 -2.968767163216418 -4.658757793419889 8.274980631669644 -5.938253824975476; 1.7555736875992336 9.282667521376071 2.07405653174241 1.5360827630811602 1.415347138010818; -2.2166629938399116 -4.817020004083972 3.669632262657891 -1.5434346026328534 -0.6591874714525724; 13.560604531822984 1.5136686735532974 -8.64066548603953 7.2083930982249464 13.650163045234647; 1.0281361434179388 -6.587606208456828 2.5188093421485584 2.659175348735485 -2.914733288425055], bias = [1.4328021361081722; 1.202403085024456; -1.2176474679140588; 0.8122208700141326; 0.05202680107995734]), layer_4 = (weight = [-2.8033417612054103 0.584138431064306 -0.019601293592080224 3.4802840445676 -1.692868112225231; 1.015918998053261 8.696745383573063 -0.220589083797818 -0.4434686844381981 1.3651228855969768], bias = [0.5579418976750097; -1.217529131811827])))




#high noise loss collection 
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
#Displaying true data vs noisy data



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
etasteps2=etasteps[1:end-10]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-10]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-10]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end


loss((layer_1 = (weight = [-0.9045435742900308 0.11238636738772709; -0.9475089625062236 -1.708825614448456; -0.8802762734726547 -4.219894387408592; 4.081873530940173 3.087696450985522; -1.8801981219467676 -2.553076798804447], bias = [-1.2343919062697963; -0.10914967665054257; -0.06665854371692435; 2.2086188662180595; -0.008212267386774403]), layer_2 = (weight = [-1.4360802799717436 -0.826953569882582 -2.2810020317468815 -1.9816757639275666 -1.760845494217153; -0.5732873781409143 -1.2249473559745303 -3.2058974751176494 -1.417039262130366 -1.5326301974636913; 1.6671193324567246 1.9360608909669377 1.3767399565214649 1.2003898086372091 0.8227904871187499; -2.003267645304169 -1.0022883992346712 -1.569297498233127 -1.368040881269627 -1.2192468675335362; 1.233288027693019 1.6689531633619459 1.4811338306125 0.9934995562718975 1.3132752399056287], bias = [-1.3281897018691537; 3.036680773110849; 1.2406601364082914; -1.5675645055778151; 1.403920775933556]), layer_3 = (weight = [1.843257008984395 4.128955319105596 1.632040930883602 1.729712711963038 0.5984816258967492; 1.7484201080074318 3.583162903539313 1.4092935984305608 1.4388136627269141 0.8184594039134077; -2.397984594287442 -1.585535742854858 -1.1353733584295234 -2.1809749974485904 -2.004499927324898; 1.7923442556253975 1.2483079866952416 0.7258531839289619 1.0943024395740166 1.280198139947649; -0.9218642515080828 -0.838280971131032 -1.3870596594083908 -1.2853957933738394 -1.059111873075892], bias = [0.2828853768745767; 0.8948272460967657; -1.5041007804823172; 0.8726472006545645; -0.3099312467610679]), layer_4 = (weight = [2.4695300621713576 -2.431542219233366 0.08198151301221193 -0.6098482241074921 -2.1005811510225745; 1.7064605169156641 -0.1591193961234026 0.7582132320819353 -0.11599385792646028 -1.2980028139858861], bias = [0.9775339067150084; -0.3735753446405967])))
