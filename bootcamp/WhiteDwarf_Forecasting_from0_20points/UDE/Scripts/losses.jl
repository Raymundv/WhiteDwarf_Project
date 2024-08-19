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
etasteps2=etasteps[1:end-80]
etaspan2 = (etasteps2[1],etasteps2[end])
prob_NN = ODEProblem(ude_dynamics,solutionarray[:,1], etaspan2, p)

#-------------------------Implementing the training routines-------------------------
eta=sol.t[1:end-80]


## Function to train the network (the predictor)

function predictude(theta, X = solutionarray[:,1], T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

training_array=solutionarray[:,1:end-80]
# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predictude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [0.44083148606258526 0.4153907613874639; -2.202925604605689 1.4505666371388228; -2.1074606303011705 1.4225765703349131; -1.1020673525581617 1.1073326755059767; 0.5757077036505083 -0.13718961814974046], bias = [-0.705582889822688; -1.270776976022611; -1.3188285404512485; -1.8391436300402724; 0.7263738307917896]), layer_2 = (weight = [0.027027245538146033 0.15076982925086596 -0.8330626081448138 -1.4691358247885977 0.9645258506455733; 2.286964491744637 1.3892888484902899 0.9233198233077189 2.0283477430111585 0.8393510300764152; -0.892193241418909 -1.2118851923167218 -1.19107323013891 -0.5825006957807912 1.8683245161738657; -0.7275961638783334 -0.21757655175808197 0.20386753726013918 -0.013319931214723477 0.9258078124320785; 0.12736637799427675 1.5542188312178697 1.5593125621162751 1.0829429789891931 2.0695354099086924], bias = [0.4744636067057505; 1.7476275508261612; -1.9505579533070396; -0.006281595920205177; -0.062295612288893173]), layer_3 = (weight = [0.47379485378695063 0.7356141547631606 0.35725964672405913 -1.9116453099897335 0.9246701048258668; 1.6760194864454876 1.9361896333911726 1.5138022017923867 1.6632739264002454 0.7296007247865723; 1.3858436467907531 -0.2506139025933762 1.2431776032475808 1.662304219724245 0.18253651896654163; 1.3063570808076213 1.6674586420905673 0.8418777024185864 0.6894448445244354 2.0153843690297673; 1.9724161183359414 1.616942420986151 1.333613551717955 1.6011264416925322 1.4694020089397495], bias = [-0.866105755543492; 1.1794554529543482; 0.952096694833618; 1.3574457325157854; 1.667364340379391]), layer_4 = (weight = [5.567278941276596e-5 -0.4205332022503853 -0.33260767248265105 -0.07367583619326605 -1.2096340307882; -4.966930948184626 -0.7535208439147982 -0.2067380261340439 -0.5477758110297998 -1.9338307521755138], bias = [-3.743399825612652e-6; 0.010559140417504246])))





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
etasteps2=etasteps[1:end-80]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)


#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-80]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-80]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [-1.515046736889739 1.4715772245693186; 1.9507636663125953 -1.81919361098221; 1.9741221996818683 -3.105294964440201; -1.9861456160038335 -1.3341902677042572; -2.3196710788532724 -1.3141409876454335], bias = [-1.7009749646935068; 1.96924157671041; 1.5840826658275742; -1.2892093947163747; -1.62795547478631]), layer_2 = (weight = [-1.435798790881446 -0.8316900125628542 -2.621142900182773 -1.7434279879516832 -1.4676339792139204; 2.0221397314834757 1.2896904786093069 1.1279923279444743 0.6056373331287483 1.8168063091135749; 1.2276024883059222 1.5273511314003574 -0.3394648956520416 0.9287920293454501 0.16159006532292622; -0.23006365041096238 0.805252974672616 -0.6202532290513828 0.19449927962559055 0.054752332478146405; 0.8597793608644825 1.2749792765350365 1.4861472782766112 0.8618851045804071 0.8958020391415344], bias = [-1.2998022689030984; 1.2149796845617464; 0.12663167074773385; 0.11531064684559203; 0.9941531984358769]), layer_3 = (weight = [1.9644601943042648 1.5555751393452593 1.7214386074199188 2.1828031625701114 1.1630394508835613; -1.4749083790428377 -1.2269016088568667 -2.1671288479950443 -2.3791057620358793 -2.819832468474865; -2.8253562552994262 -2.5253283406358484 -2.247845776437411 -2.9588069378969912 -2.9559184852763627; 1.8235031874765784 0.8066004884151378 1.1700424233636308 1.2095351916379218 1.34988273982248; -1.005832355106733 -1.5626995845676028 -1.5886445244325216 -1.595982580866825 -1.4766526613506339], bias = [1.604507356776649; -2.285230568154275; -2.564351269966508; 1.4023711470007345; -1.441890740374194]), layer_4 = (weight = [-0.37588249359197706 -1.4681420713897797 -1.0097507230318084 0.28950046371967686 -0.709842213486935; -1.0084995840033157 -1.5228515612249625 -1.1729265656229195 -0.17645659041616316 -0.7864599917335807], bias = [0.02241619536226371; -0.8038770606255771])))

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
etasteps2=etasteps[1:end-80]
etaspan2 = (etasteps2[1],etasteps2[end])


# Defining the UDE problem
prob_NN = ODEProblem(ude_dynamics,I, etaspan2, p)


#-------------------------Implementing the training routines-------------------------



## Function to train the network (the predictor)
eta=sol.t[1:end-80]
function predict_ude(theta, X = I, T = eta)
    _prob = remake(prob_NN, u0 = X, tspan = (T[1], T[end]), p = theta)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

#Training Array
training_array=x1_noise[:,1:end-80]

# Defining the L2 loss, that will be minimized
function loss(theta) 
    X̂ = predict_ude(theta)
    sum(abs2, training_array .- X̂)
end

loss((layer_1 = (weight = [-1.8193159452164203 2.744073559928421; 1.8114828090663073 -2.824422799463003; -1.1871640793049103 0.1991819345565677; -1.9797832626046368 -1.274215682796942; -2.9797708415831488 -0.5704117339668496], bias = [-1.9753779669633234; 1.8315393640754296; -1.819746187913108; -1.2626804101770939; -2.157606056595257]), layer_2 = (weight = [-1.656996600323156 -1.0219355838381443 -2.4718419206721456 -1.9329616455713972 -1.590472057068167; 2.2899365632163615 1.5408609121100554 1.3239409821240542 0.8208608706629149 1.9876863575368522; 0.7363991094295163 1.305363764437111 -0.6198462602084744 0.7490730937937635 -0.15184425974692353; -0.012629502432272646 0.8725598564012396 -0.6364575074541449 0.23592389370334638 0.058425763341647126; 0.9121718146070315 1.2519985935044466 1.2542535069996472 0.819289787390472 0.7079388404595489], bias = [-1.623987413618985; 1.5413492780932259; -0.03380050160553218; 0.036630168215904216; 0.6782680019010362]), layer_3 = (weight = [1.9759394170174651 1.5395373068511884 1.7051080679882484 2.1932001329092166 1.195254885897714; -1.59460929730924 -1.0080664788216198 -2.3001334835338882 -2.334234455912098 -2.550182176191753; -2.554137033554578 -1.7721260902612217 -2.0789249193115116 -2.777366494470448 -2.7829394649623036; 1.7240012209562932 0.7872925611487056 1.5973228673486781 1.2533588035800738 1.1714516173926828; -0.9949778922259092 -1.52495466337609 -1.5605743559688732 -1.5867058292853926 -1.4617828914896809], bias = [1.6111057626333802; -2.2011852385946535; -2.235313483555916; 1.303586508751524; -1.4244039633051124]), layer_4 = (weight = [0.2868896085245463 -1.3326704793504212 -0.9387036268390881 0.5881310032855239 -0.4237699378572846; -0.9862025525458352 -1.5063592971174946 -1.1699786455554597 -0.0042641372604034205 -0.772597586415812], bias = [0.1539669615205976; -0.7738596664851816])))