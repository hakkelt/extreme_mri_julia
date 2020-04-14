try
    using CuArrays # we have CUDA, so this should not fail
catch ex
    # something is wrong with the user's set-up (or there's a bug in CuArrays)
    @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
end
try
    using CUDAnative # we have CUDA, so this should not fail
catch ex
    # something is wrong with the user's set-up (or there's a bug in CUDAnative)
    @warn "CUDA is installed, but CUDAnative.jl fails to load" exception=(ex,catch_backtrace())
end
try
    using CUDAdrv # we have CUDA, so this should not fail
catch ex
    # something is wrong with the user's set-up (or there's a bug in CUDAdrv)
    @warn "CUDA is installed, but CUDAdrv.jl fails to load" exception=(ex,catch_backtrace())
end

## -------------------------------------
## ------- fftshift bugfix -------------
## -------------------------------------

function copyto!(dest::CuArray{T}, d_range,
                      src::CuArray{T}, s_range) where T
    len = length(d_range)
    if length(s_range) != len
        throw(ArgumentError("Copy range needs same length. Found: dest: $len, src: $(length(s_range))"))
    end
    len == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    Base.copyto!(dest, d_offset, Base.materialize(src), s_offset, len)
end

function copyto!(dest::AbstractArray{T}, d_range,
                      src::AbstractArray{T}, s_range) where T
    Base.copyto!(dest, d_range, src, s_range)
end

@noinline function circshift!(dest::AbstractArray{T,N}, src, shiftamt::Base.DimsInteger) where {T,N}
    dest === src && throw(ArgumentError("dest and src must be separate arrays"))
    inds = axes(src)
    axes(dest) == inds || throw(ArgumentError("indices of src and dest must match (got $inds and $(axes(dest)))"))
    _circshift!(dest, (), src, (), inds, Base.fill_to_length(shiftamt, 0, Val(N)))
end
circshift!(dest::AbstractArray, src, shiftamt) = circshift!(dest, src, (shiftamt...,))

@inline function _circshift!(dest, rdest, src, rsrc,
                             inds::Tuple{AbstractUnitRange,Vararg{Any}},
                             shiftamt::Tuple{Integer,Vararg{Any}})
    ind1, d = inds[1], shiftamt[1]
    s = mod(d, length(ind1))
    sf, sl = first(ind1)+s, last(ind1)-s
    r1, r2 = first(ind1):sf-1, sf:last(ind1)
    r3, r4 = first(ind1):sl, sl+1:last(ind1)
    tinds, tshiftamt = Base.tail(inds), Base.tail(shiftamt)
    _circshift!(dest, (rdest..., r1), src, (rsrc..., r4), tinds, tshiftamt)
    _circshift!(dest, (rdest..., r2), src, (rsrc..., r3), tinds, tshiftamt)
end
# At least one of inds, shiftamt is empty
function _circshift!(dest, rdest, src, rsrc, inds, shiftamt)
    copyto!(dest, CartesianIndices(rdest), src, CartesianIndices(rsrc))
end

## -------------------------------------
## --------- Apodizations --------------
## -------------------------------------

function _apodize!(signal::CuArray, apodized_dims, oversamp, width, β)

    all_dims = ndims(signal)
    untouched_dims = all_dims - apodized_dims
    for axis in range(untouched_dims + 1, all_dims, step=1)
        axis_size = size(signal, axis)
        oversampled_size = ceil(oversamp * axis_size)

        # Calculate apodization window
        recip_iFFT_Kaiser_Bessel_kernel(x) = begin
            tmp = √(β^2 - (π * width * (x - axis_size ÷ 2) / oversampled_size)^2)
            tmp /= sinh(tmp)
            tmp
        end
        window = cu(recip_iFFT_Kaiser_Bessel_kernel.(0:axis_size-1))

        # Apply point-wise along selected axis, broadcast along all other dimensions
        broadcast_shape = ones(Int, all_dims)
        broadcast_shape[axis] = axis_size
        signal .*= reshape(window, broadcast_shape...)
    end
    return signal
end

## -------------------------------------
## ------------ Gridding ---------------
## -------------------------------------

@generated function _cu_gridding_kernel(threaded_output, input, coord, kernel, width, ::Val{ndim}) where ndim
    quote
        block_index = blockIdx().x
        block_stride = gridDim().x
        for point_index = block_index:block_stride:size(coord, 1)
            output_shape = size(threaded_output)
            @nextract($ndim, n, d -> output_shape[$ndim-d+3]) # (2, batch size, dims...)
            @nextract($ndim, interval_middle,
                d -> coord[point_index, $ndim - d + 1])
            @nextract($ndim, interval_start,
                d -> ceil(Int, interval_middle_d - width / 2))
            @nextract($ndim, interval_end,
                d -> floor(Int, interval_middle_d + width / 2))
            range = @ntuple($ndim, d -> interval_start_d:interval_end_d)
            thread_index = threadIdx().x - 1
            thread_stride = blockDim().x
            for i in thread_index:thread_stride:width^$ndim-1
                # linear index to tuple of subscripts and shift to window start:
                @nextract($ndim, idx, d -> i ÷ width^(d-1) % width + interval_start_d)
                # positions in kernel vector:
                @nextract($ndim, location, d -> abs(idx_d - interval_middle_d) / (width / 2))
                # linear interpolation:
                w = @ncall($ndim, *, d -> lin_interpolate(kernel, convert(eltype(kernel), location_d)))
                position = @ntuple($ndim, d -> (idx_{$ndim-d+1} + n_{$ndim-d+1}) % n_{$ndim-d+1} + 1)
                #if point_index < 2
                #    @cuprintln(position[1], position[2], position[3])
                #end
                for b in 1:size(input, 1)
                    v = w * input[b, point_index]
                    @atomic threaded_output[1, b, position...] += convert(eltype(threaded_output), v.re)
                    @atomic threaded_output[2, b, position...] += convert(eltype(threaded_output), v.im)
                end
            end
        end
    end
end

function _cu_gridding!(
        input::CuArray{Complex{T}, 2},
        coord::CuArray{T, 2},
        output_shape::NTuple{D, Int},
        kernel::CuArray{T, 1},
        width::Int) where {T, D}
    ndim = size(coord, 2)
    num_threads = width^ndim
    how_many_can_be_used = size(coord, 1)
    num_blocks = min(4096, how_many_can_be_used)
    threaded_output = CuArrays.zeros(eltype(coord), (2, output_shape...))
    @cuda(threads=num_threads, blocks=num_blocks,
        _cu_gridding_kernel(threaded_output, input, coord, kernel, width, Val(ndim)))
    output = similar(input, output_shape...)
    output .= threaded_output[1, ..] .+ threaded_output[2, ..] .* 1im
    CuArrays.unsafe_free!(threaded_output)
    return reshape(output, output_shape)
end

function _gridding(
        input::CuArray{Complex{T}, 2},
        coord::CuArray{T, 2},
        output_shape::NTuple{D, Int},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    _cu_gridding!(input, coord, output_shape, cu(kernel), width)
end

function _gridding(
        input::CuArray{T, 2},
        coord::CuArray{T, 2},
        output_shape::NTuple{D, Int},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    _cu_gridding!(input, coord, output_shape, cu(kernel), width)
end
