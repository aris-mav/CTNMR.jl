using LinearAlgebra
using NMRInversions
using GLMakie
using Optimization, OptimizationOptimJL
using Serialization

function read_raw_data(filename = "Berea_2d25um_binary.raw")

    data = zeros(Bool,1000,1000,1000)

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
    
    return data 
end

# Pores are 0, matrix is 1
# CT_data is true for solid space and ~CT_data is true for porespace
#=porosity =  count(.!data) / length(data)=#

function run_random_walk(
    data;
    n_walkers = 1000,
    n_steps = 10000,
    relaxivity = 20e-6, #m/s
    D = 2.96e-9, #(water, m^2 s^-1)
    voxel_length = 2.25e-6 , # μm e-6 (m)
    step_length = voxel_length/8,
)

    # Find all indices of pore spaces, and select some of them to 
    # use as starting points, one for each walker
    starting_point_indx = rand(findall(iszero, data), n_walkers);

    # Convert to xyz coordinates 
    xyz = [collect(Tuple.(starting_point_indx[i])) for i in 1:n_walkers] .* voxel_length

    # select random position within voxel
    xyz = xyz  .- [rand(3) for _ in 1:n_walkers] .* voxel_length;

    ## Loop
    kill_probability = (2*relaxivity*step_length) / (3*D)
    M = zeros(Int, n_steps)

    for i = 1:n_steps

        #=if i % 1000 == 0=#
        #=    @show i=#
        #=end=#

        xyz_step = [LinearAlgebra.normalize(randn(3)) * step_length for _ in 1:length(xyz)]

        # Find which walkers hit a wall
        hitwall = [data[ ceil.(Int, i./voxel_length)...] for i in (xyz .+ xyz_step)]

        # Step, only for those who are not about to hit a wall and are alive 
        # (others are just left where they are)
        xyz[.~hitwall] .= xyz[.~hitwall] .+ xyz_step[.~hitwall];

        # Decide which walkers will die
        for (i,x) in enumerate(hitwall)
            if x && rand() < kill_probability
                popat!(xyz,i)
            end
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

    println("testing for relaxivity $(u[1])")
    t, M = run_random_walk(data, n_steps = Int(6e5), relaxivity = u[1])

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

    return sum((M_compressed .- real(exp_data.y)).^2)

end

function find_relaxivity(ct_data, exp_data)

    t_compressed = zeros(length(exp_data.x));
    M_compressed = zeros(length(exp_data.x));

    p = Optimization.OptimizationProblem(
        Optimization.OptimizationFunction(cost), 
        [20e-6],
        (ct_data, t_compressed, M_compressed, exp_data),
        lb=[1e-8], ub=[1e-5]
    )
    ρ = OptimizationOptimJL.solve(p, Optim.ParticleSwarm(), maxiters=50, maxtime=1e5)

    return ρ
end

data = read_raw_data();
exp_data = deserialize("./exp_data.bin");
find_relaxivity(data, exp_data)
