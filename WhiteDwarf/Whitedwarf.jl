#Loading the necessary libraries
using Plots
using DifferentialEquations
using Random
using Statistics
rng = Random.default_rng()
Random.seed!(99)

#Constants
C = 0.01


#Initial Conditions
I = [1, 0]
etaspan = (0.05, 5.325)

#Define the problem
function whitedwarf(du, u, p, r)
    psi = u[1]
    dpsi = u[2]
    du[1] = dpsi
    du[2] = (-((psi^2-C))^(3/2) - 2/r * dpsi)
end


#Pass to solvers
prob = ODEProblem(whitedwarf, I, etaspan)
sol = solve(prob, Tsit5())

#Plot
plot(sol, idxs = (0, 1), linewidth = 1, title = "White Dwarf equation", xaxis = "\\eta",
    yaxis = "\\phi", label = "\\phi")

#--------------I will solve the white dwarf equation using the SecondOrderODEProblem

function whitedwarf2(ddu,du,u,C,eta)
    ddu .= (-((u^2-C))^(3/2) - 2/eta * du)
end


dpsi0=0
psi0=1
prob2 = SecondOrderODEProblem(whitedwarf2,dpsi0, psi0, etaspan, C)
sol2 = solve(prob)
eta = sol.t
#plot
plot!(sol2,idxs = (0, 1), linewidth=1.5, title = "White Dwarf equation", xaxis = "\\eta", yaxis="density", label = "\\phi")

#Adding noise to data:

x1=Array(sol)

x1_mean = mean(x1, dims = 2)
noise_magnitude = 5e-2
x1_noise = x1 .+ (noise_magnitude*x1_mean) .* randn(eltype(x1), size(x1))

plot(sol, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(eta, transpose(x1_noise), color = :red, label = ["Noisy Data" nothing])
