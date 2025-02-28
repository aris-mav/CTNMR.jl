function count_surface_points(A, voxel_length)

    A[1,:,:] .= false
    A[:,1,:] .= false
    A[:,:,1] .= false
    A[end,:,:] .= false
    A[:,end,:] .= false
    A[:,:,end] .= false

    neighbor_indices = (
        CartesianIndex(1, 0, 0),
        CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, 1),
        CartesianIndex(-1, 0, 0),
        CartesianIndex(0, -1, 0),
        CartesianIndex(0, 0, -1),
    )

    pore_counter = 0
    surface_counter = 0

    for i in CartesianIndices(A)

        A[i] == false && continue
        pore_counter += 1

        for n in neighbor_indices
            if !A[i .+ n]
                surface_counter += 1
            end
        end

    end
    return (surface_counter * voxel_length^2) / (pore_counter * voxel_length^3)
end

#=@time count_surface_points(rand(Bool,100,100,100),0.01)=#
