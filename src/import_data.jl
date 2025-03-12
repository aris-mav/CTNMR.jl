export read_raw_data
function read_raw_data(filename::String, dimensions::Tuple{Int64, Int64, Int64})

    println("Reading data from : " * filename)
    data = zeros(UInt8, dimensions...)

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

    println("Data read successfully.")

    porosity = count(data .!= grain) / length(data)
    @show porosity # sanity check

    if  maximum(data) == 2
        CO2fraction = count(data .== CO2) / length(data)
        @show CO2fraction
    end

    flush(stdout)
    
    return data 
end


export read_vox_size
function read_vox_size(file)
    x = open(file*".dict") do io
        readuntil(io,"delta.x=")
        x = parse(Float64, readline(io))
        return x
    end
end
