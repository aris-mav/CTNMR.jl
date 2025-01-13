using LinearAlgebra
using NMRInversions
using Optim
using UnicodePlots
using StaticArrays

function read_raw_data(filename::String)

    println("reading data from " * filename)
    data = zeros(UInt8, 1600,1600,1100)

    open(filename, "r") do io
        i = 1
        while !eof(io)
            data[i] = read(io, UInt8)
            i += 1
        end
    end
    
    grain::UInt8 = 1
    brine::UInt8 = 0
    CO2::UInt8 = 9
    if  maximum(data) == 2
        grain = 2
        brine = 1
        CO2 = 0
    end

    # Make edges solid, so that the walkers cannot escape
    data[1,:,:].= grain;
    data[:,1,:].= grain;
    data[:,:,1].= grain;
    data[end,:,:].= grain;
    data[:,end,:].= grain;
    data[:,:,end].= grain;

    println("data reading success")

    porosity = count(data .!= grain) / length(data)
    @show porosity # sanity check
    
    return data 
end


function run_random_walk(lattice;
                         relaxivity::Float64 = 20e-6, #m/s
                         n_walkers::Int= 10^4, # per thread
                         n_steps::Int = 10^4,
                         D::Float64 = 2.96e-9, #(water, m^2 s^-1)
                         voxel_length::Float64 = 2.25e-6 , # μm e-6 (m)
                         step_length::Float64 = voxel_length/7, 
                         n_threads::Int = Threads.nthreads()
                         )

    grain::UInt8 = 1
    brine::UInt8 = 0
    CO2::UInt8 = 9
    if  maximum(lattice) == 2
        grain = 2
        brine = 1
        CO2 = 0
    end

    n_died = [zeros(Int, n_steps) for _ in 1:n_threads] # number that died on step i
    valid_starting_points = findall(x -> x != grain , lattice);
    kill_probability = (2*relaxivity*step_length) / (3*D)

    Threads.@threads for i in 1:n_threads

        for _ in 1:n_walkers

            xyz = SVector{3, Float64}(Tuple(rand(valid_starting_points)) .* voxel_length)
            xyz -= (@SVector rand(3)) .* voxel_length

            for t in 1:n_steps
                xyz += (step = normalize(@SVector(randn(3))) * step_length)
                voxel = lattice[Int.(cld.(xyz , voxel_length))...]

                if voxel == grain
                    if rand() < kill_probability
                        n_died[i][t] += 1
                        break
                    else # take a step back and continue
                        xyz -= step
                    end
                elseif voxel == CO2 # take a step back and continue
                    xyz -= step
                end
            end
        end
    end

    time_step = step_length^2 / 6D
    t = collect(1:n_steps) * time_step
    M = (n_walkers * n_threads) .- cumsum(sum(n_died))

    return t,M
end


function cost(u,p)

    data = p[1];
    t_compressed = p[2];
    M_compressed = p[3];
    exp_data = p[4];
    vox_l = p[5]

    t, M = run_random_walk(data, 
                           relaxivity = u[1], 
                           n_steps = Int(6e5), 
                           n_walkers = 1000 , 
                           voxel_length = vox_l)

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

