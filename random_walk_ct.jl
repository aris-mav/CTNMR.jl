using LinearAlgebra
using NMRInversions
using Optim
using UnicodePlots
using Random

function read_raw_data(filename::String)

    println("reading data from " * filename)
    data = zeros(UInt8,1600,1600,1100)

    open(filename, "r") do io
        i = 1
        while !eof(io)
            data[i] = read(io, UInt8)
            i += 1
        end
    end
    
    # Make edges solid, so that the walkers cannot escape
    data[1,:,:].=true;
    data[:,1,:].=true;
    data[:,:,1].=true;
    data[end,:,:].=true;
    data[:,end,:].=true;
    data[:,:,end].=true;

    println("data reading success")
    
    return data 
end

# Pores are 0, matrix is 1
# CT_data is true for solid space and ~CT_data is true for porespace
#=porosity =  count(.!data) / length(data)=#

function run_random_walk(
    data;
    n_walkers = 10^4,
    n_steps = 10^4,
    relaxivity = 20e-6, #m/s
    D = 2.96e-9, #(water, m^2 s^-1)
    voxel_length = 2.25e-6 , # μm e-6 (m)
    step_length = voxel_length/10,
)

    grain = 1
    brine = 0
    three_phases = maximum(data) == 2

    if three_phases 
        grain = 2
        brine = 1
    end
    CO2 = 0

    Random.seed!(1)

    # Find all indices of pore spaces, and select some of them to 
    # use as starting points, one for each walker
    starting_point_indx = rand(findall(x -> x == brine, data), n_walkers);

    # Convert to xyz coordinates 
    xyz = [collect(Tuple(starting_point_indx[i])) for i in 1:n_walkers] .* voxel_length

    # select random initial position within voxel
    xyz = xyz  .- [rand(3) for _ in 1:n_walkers] .* voxel_length

    ## Loop
    kill_probability = (2*relaxivity*step_length) / (3*D)
    M = zeros(Int, n_steps)

    for i = 1:n_steps

        xyz_step = [LinearAlgebra.normalize(randn(3)) * step_length for _ in eachindex(xyz)]

        # Find which walkers hit a wall
        hitwall = [data[ ceil.(Int, position./voxel_length)...] == grain for position in (xyz .+ xyz_step)]

        hitCO2 = zeros(Bool, length(xyz))
        if three_phases
            hitCO2 = [data[ ceil.(Int, position./voxel_length)...] == CO2 for position in (xyz .+ xyz_step)]
        end

        hitsomething = hitwall .|| hitCO2

        # Step, only for those who are not about to hit a wall or CO2
        # (others are just left where they are)
        xyz[.~hitsomething] .= xyz[.~hitsomething] .+ xyz_step[.~hitsomething]

        # Kill those who hit a wall and don't pass the probability test
        # reverse so that it pops bottom to top and then we don't have to worry
        # about indexing out of bounds, since the array can shrink
        for i in reverse(findall(hitwall))[rand(count(hitwall)) .< kill_probability]
            popat!(xyz, i)
        end

        M[i] = length(xyz)
    end

    time_step = step_length^2 / 6D
    t = collect(1:n_steps) * time_step

    return t, M
end


function cost(u,p)

    data = p[1];
    t_compressed = p[2];
    M_compressed = p[3];
    exp_data = p[4];
    vox_l = p[5]

    t, M = run_random_walk(data, n_steps = Int(6e5), relaxivity = u[1], voxel_length = vox_l)

    if exp_data.seq in [NMRInversions.IR]
        M = 1 .- 2 .* M ./ maximum(M)
    else
        M = M ./ maximum(M)
    end

    for (i, x) in enumerate(exp_data.x)
        ind = argmin(abs.(x .- t))
        t_compressed[i] = t[ind]
        M_compressed[i] = M[ind]
    end

    data_y = real.(exp_data.y) ./ maximum(real(exp_data.y))

    residuals = M_compressed .- data_y
    cost = norm(residuals, 1)

    p = lineplot(t_compressed, M_compressed, name = "Simulation",
                 title = "Rho: $(u[1]), Cost: $(cost)",xscale=:log10);
    lineplot!(p, exp_data.x, data_y , name = "Experiment")

    println(p)

    return cost

end

function find_relaxivity(ct_data, exp_data, vox_l)

    t_compressed = zeros(length(exp_data.x));
    M_compressed = zeros(length(exp_data.x));

    ρ = optimize(
        x -> cost(x, (ct_data, t_compressed, M_compressed, exp_data, vox_l)),
        1e-5, 1e-3
    )

    return ρ
end

function read_vox_size(file)
    x = open(file*".dict") do io
        readuntil(io,"delta.x=")
        x = parse(Float64, readline(io))
        return x
    end
end

