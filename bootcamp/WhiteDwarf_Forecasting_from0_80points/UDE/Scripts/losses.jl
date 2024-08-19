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
etasteps2=etasteps[1:end-20]
etaspan2 = (etasteps2[1],etasteps2[end])
prob_NN = ODEProblem(ude_dynamics,solutionarray[:,1], etaspan2, p)

#-------------------------Implementing the training routines-------------------------
eta=sol.t[1:end-20]


## Function to train the network (the predictor)

function predictude(theta, X = solutionarray[:,1], T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

training_array=solutionarray[:,1:end-20]
# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predictude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [-1.1143660766262544 -0.36611705855786186; -1.7519205534665814 -0.9109177462877476; -2.3642338377310965 1.3086370454181933; -0.6237617892634615 -1.5179773305727846; 1.523911577785339 -0.1897973780519804], bias = [-0.5465130224095108; -0.6415232773305236; -1.8544952620545587; -0.4680265047804734; 1.1368763014459187]), layer_2 = (weight = [0.12376415833635131 0.5976282708541546 -0.8881743065557622 -1.1591050508613434 -0.25284342524178427; 2.0656201734983606 1.4440180171847894 0.9356753982359111 1.8576049517374833 0.9871442440770988; -0.8069096950792849 -0.7790407419085087 -0.6264272874865515 -0.08727777223592115 -0.5969660964482135; 1.7281110815784764 1.0673078718364652 0.6142548005100602 1.7603324800093139 1.0611792212163953; 1.307155425536457 2.0090716491173044 1.7315241121597185 1.7247208586504215 1.6046595429544215], bias = [0.2523530085430899; 1.6052302609568787; -0.849801513237853; 1.8070626464615305; -0.10325025597845644]), layer_3 = (weight = [-0.7405831226452029 0.17012316882186057 0.325948183654489 -0.7703516708391075 1.084989132777383; 1.7597565115569025 1.93611543000045 1.5523286169185215 1.6874118424008855 0.7308363447680933; 0.953177488885703 0.5079633715580499 1.1426492034570124 1.3013546840682977 0.599886391707636; 0.8222600720131266 1.46351795870213 0.46252840054134947 0.31385389652957346 2.0147749572433136; 1.746504815591536 1.4500111280971937 1.156104223424472 1.504885947571245 1.1087962117562735], bias = [-0.2611212917982521; 1.3001892025668478; 0.3023587271577198; 0.5800854952329786; 1.615436123621225]), layer_4 = (weight = [0.0017050674089267661 -0.29077379246187257 0.006797856167955862 -0.003813275554236909 -1.1183038703779917; -0.8985411166986927 -0.5476497611743479 1.4250797666819106 0.7359261702101261 -1.4376148845900374], bias = [-0.0017380786259726926; -0.10251374553146454])))

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
etasteps2=etasteps[1:end-20]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-20]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-20]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [-2.7250261860223994 -5.030143394739985; -5.059239219219269 1.2775870131418356; 1.635871887273583 -16.811760241138913; -6.472797895037545 -7.605136358131178; 0.05362238451391332 -2.379862215283128], bias = [-1.4505878306698454; -11.389025222030613; -6.968085657430716; 0.44784345625888755; -4.509712654975539]), layer_2 = (weight = [-1.5644065895067432 -0.9164755611509446 -2.5265464929400383 -1.998589591285494 -2.361101054532156; 6.086103805304516 1.806967965495118 4.895288061656784 -0.8764168618703745 -2.013823162163969; 0.8027830327045455 0.5078779881680192 -2.543847638405015 2.3501328903805514 -6.72191713239917; -2.1693394150369034 -1.0723514805136851 -1.967485662853939 -0.9700017379655125 -1.6727854944771; 1.3358018426773954 1.6916623777397022 1.6817403824722732 1.177807488941168 1.4908127816679069], bias = [-2.1032972918626576; 0.06858984046529645; 1.0124974951622834; -3.9151617549547932; 2.691993451281844]), layer_3 = (weight = [1.9191413958699335 2.33721202450939 2.5774728704439998 1.4281448363440992 0.3086434391160802; 1.7194759243867517 2.623380777933382 1.4243627120606202 1.315059446099901 0.8214980271140254; -2.373219357720962 -3.7496820268564854 0.6758231452553451 -2.074883711699629 -2.0239359739125358; 1.6982784909162463 2.1309263130595313 0.8818962087917394 0.9386533122222048 1.2129933205088093; -0.9217498708274439 -0.4662292997768613 -1.0614653989626823 -1.2087979991917976 -1.0502574207067565], bias = [-1.0426941034930575; 2.8218089345382413; -2.138128192388243; 2.365454810499295; -3.2837272574640894]), layer_4 = (weight = [-0.0475313299240088 -0.8411893324775517 0.4152477807418682 -1.5088226476139852 3.46991584415588; 4.403913669421526 -0.8326075948657466 0.5850608417913736 1.3271292315404777 1.1453405510233599], bias = [0.0012513852493203767; -0.8776590219802635])))


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
etasteps2=etasteps[1:end-20]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)




#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-20]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-20]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [-4.693735572274568 -20.230549713732305; 1.0250946705472488 -2.8708184084905115; -0.7130547069742962 -12.098650901489295; -1.2826738697230091 1.1003366594225643; -9.905843963447074 -11.317112889587166], bias = [-3.6609834001055113; -2.350318095961985; -1.883904499982611; -2.1468423905882856; 1.9201222852828537]), layer_2 = (weight = [-2.4109251565839527 -2.288479840822751 -5.837684864870527 -2.002693181094623 -3.3124303590717625; 1.8080736001552369 -3.1076518964667534 1.5914280512166277 -0.24614081508911242 16.886000122421954; 3.5901332892919515 3.083091661367663 6.9415341934220764 1.3302670125285887 2.990179439881339; -1.8210309541262735 -1.503145209570274 -1.7100822209432072 -0.9747876314979352 -1.5955715147626897; 0.7690991175758467 6.07421772313102 4.824358356061422 1.0816428224240664 1.7095341764186578], bias = [0.6599019990717282; 0.625396688073301; -2.709458401447426; -3.300105671566056; -2.543757690012234]), layer_3 = (weight = [2.3466100558658733 5.88727953103011 -0.740440267032653 1.417410921457007 -0.9686074587203708; 1.3834695597922828 9.577401142058136 1.2479222411727953 1.2405606999501293 -0.034903683014875064; -2.374236979958666 -0.9854477158969225 -1.1349751861036776 -2.0721207335808494 -2.01448006710596; 2.645041492666396 -3.047731041888801 7.980438904733448 0.8700331102872829 6.497553481066188; -0.6730417649868091 -11.795172588903215 -3.700567783206516 -1.3931933348827144 0.5751027937129929], bias = [0.7495575461748087; 0.9642492713810848; -3.2697912000023175; 1.0570285140028801; -1.0832671767525202]), layer_4 = (weight = [13.88451943087403 -7.683833863189535 -1.581169344091102 -7.299215583594375 -9.906861313941544; -3.9837447113080757 3.171384757387331 0.824158100167602 0.09398098535887177 4.915227709649447], bias = [0.4670884933629055; -0.5732678497329504])))
