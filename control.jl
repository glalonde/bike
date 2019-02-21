using Revise
using StaticArrays
using GeometryTypes
using OrdinaryDiffEq

const Vec5 = SVector{5, Float64}
const Vec4 = SVector{4, Float64}
const Vec3 = SVector{3, Float64}
const Vec2 = SVector{2, Float64}
const AlignedBox2d = HyperRectangle{2, Float64}
const AlignedBox4d = HyperRectangle{4, Float64}
const AlignedBox5d = HyperRectangle{5, Float64}


# Model of a bicycle where the traction wheel is the rear, and the steered wheel is the front.
# State dimension is [x, y, orientation, steering angle, traction velocity]
# Control dimension
struct Bicycle
    wheelbase::Float64
    # [steering angle, traction velocity, steering velocity, traction acceleration]
    state_space::AlignedBox5d
    control_space::AlignedBox2d
    control_cycle::Float64
end

function BicycleDynamics(plant::Bicycle, x::Vec5, u::Vec2)
    # of the traction wheel
    turning_radius = plant.wheelbase / tan(x[3])
    translational_velocity = x[5]
    angular_velocity = x[5] / turning_radius
    return Vec5(cos(x[3]), sin(x[3]), angular_velocity, u[1], u[2])
end

function Simulate(plant::Bicycle, init::Vec5, control_sequence::T) where {T <: AbstractArray{Vec2, 1}}
    function integration_function(u,p,t)
        control_step = clamp(trunc(Int, t / plant.control_cycle) + 1, 1, length(control_sequence))
        return BicycleDynamics(plant, u, control_sequence[control_step])
    end
    problem = ODEProblem(integration_function, init, (0.0, plant.control_cycle * length(control_sequence)))
    return solve(problem, Tsit5())
end

function BoxFromExtents(min::SVector{S,T}, max::SVector{S,T}) where {S, T}
    origin = (min + max) / T(2)
    sizes = (max - min) / T(2)
    return HyperRectangle{S, T}(origin, sizes)
end

function MakeBike()
    max_speed = 10.0
    max_acceleration = 1.0
    max_steering_velocity = 1.0
    max_steering_angle = π / 3
    wheelbase = 1.0
    max_state_vec = Vec5(Inf, Inf, Inf, max_steering_angle, max_speed)
    max_control_vec = Vec2(max_steering_velocity, max_acceleration)
    dt = .1
    return Bicycle(wheelbase, BoxFromExtents(-max_state_vec, max_state_vec), BoxFromExtents(-max_control_vec, max_control_vec), dt)
end

bike = MakeBike()
init = Vec5(0, 0, 0, 0, 0)
control_sequence = Vec2[]
push!(control_sequence, Vec2(1, 1))
Simulate(bike, init, control_sequence)