module SigJl

using Documenter, EllipsisNotation, FFTW, PaddedViews, SpecialFunctions, Base.Cartesian, Setfield, LinearAlgebra, ProgressMeter

using CUDAapi # this will NEVER fail
if has_cuda()
    #include("CuSigPy.jl")
end

export nufft, nufft_adjoint

mutable struct fftPlanHolder
    plan::Union{AbstractFFTs.Plan, Nothing}
end

mutable struct griddingBufferHolder
    buffers::Union{Array, Nothing}
end

struct NUFFT_plan{T}
    coord::AbstractArray{T}
    img_temp::SubArray
    oversampled_img_temp::AbstractArray{Complex{T}}
    oversampled_img_temp2::AbstractArray{Complex{T}}
    kernel::AbstractArray{T}
    FFTplan::fftPlanHolder
    iFFTplan::fftPlanHolder
    griddingBuffer::griddingBufferHolder
    oversamp::Real
    width::Int
    n::Int
    β::T
    adjoint::Bool
    progressTitle::Union{String, Nothing}
    progress_dt::Real
end

Base.adjoint(plan::NUFFT_plan) = @set plan.adjoint = !plan.adjoint

function nufft_plan(
        coord::AbstractArray{T},
        img::Union{AbstractArray{Complex{T}}, NTuple{N, Int}, Tuple, Nothing} = nothing;
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {N, T}
    
    if img isa AbstractArray
        img_shape = size(img)
    elseif img isa NTuple && eltype(img) <: Int
        img_shape = img
    elseif img isa Tuple
        @assert(all(x -> x isa Int, img[1:end-1]) && img[end] isa EllipsisNotation.Ellipsis, 
            "You must either specify all dimensions of the image as a tuple of Ints,
                or you must specify the batch dimensions, and append an ellipsis mark ('..')
                to denote the unknown image dimensions.")
        img_shape = tuple(img[1:end-1]..., estimate_shape(coord)...)
    else
        img_shape = estimate_shape(coord)
    end
    
    ndim = size(coord, 1)
    all_dims = length(img_shape)
    @assert(all_dims ≥ ndim,
        "The size of coord along the last dimension should be greater or equal then the dimensionality of input.")
    
    β = convert(eltype(coord), π * √(((width / oversamp) * (oversamp - 0.5))^2 - 0.8))
    oversampled_shape = _get_oversamp_shape(img_shape, ndim, oversamp)
    shift = oversampled_shape .÷ 2 .- img_shape .÷ 2 .+ 1
    view_range = UnitRange.(shift, shift .+ img_shape .- 1)
    
    oversampled_img_temp = similar(coord, Complex{eltype(coord)}, oversampled_shape)
    img_temp = @view oversampled_img_temp[view_range...] # used for implicit cropping and zero padding
    oversampled_img_temp2 = similar(coord, Complex{eltype(coord)}, oversampled_shape)
    
    coord = _scale_coord(coord, img_shape, oversamp)
    
    x = range(0, stop=n-1, step=1) ./ n
    kernel = convert.(eltype(coord), window_kaiser_bessel.(x, width, β))
    
    NUFFT_plan(coord, img_temp, oversampled_img_temp, oversampled_img_temp2, kernel,
        fftPlanHolder(nothing), fftPlanHolder(nothing), griddingBufferHolder(nothing),
        oversamp, width, n, β, false, progressTitle, progress_dt)
end

function nufft_plan(
        coord::AbstractArray{T},
        img::AbstractArray{T};
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {N, T}
    nufft_plan(coord, convert.(Complex{eltype(img)}, img), oversamp=oversamp, width=width, n=n, progressTitle=progressTitle, progress_dt=progress_dt)
end

function _nufft_forw(ksp::AbstractArray{Complex{T}}, plan::NUFFT_plan{T}, img::AbstractArray{Complex{T}}) where T
    
    ndim = size(plan.coord, 1)
    fill!(plan.oversampled_img_temp, zero(eltype(img))) # initialize oversampled buffer
    
    if !(plan.progressTitle isa Nothing)
        (p = Progress(100, plan.progress_dt, plan.progressTitle))
    else
        p = nothing
    end
    
    # Apodize
    plan.img_temp .= img # Note that img_temp is a view to the centered region of oversampled_img_temp!
    _apodize!(plan.img_temp, ndim, plan.oversamp, plan.width, plan.β)
    img_shape = size(img)
    plan.img_temp ./= convert(eltype(img), √(prod(img_shape[end-ndim+1:end])))
    
    !(plan.progressTitle isa Nothing) && next!(p)

    # FFT
    all_dims = ndims(img)
    dims = tuple((all_dims-ndim+1:all_dims)...)
    ifftshift!(plan.oversampled_img_temp2, plan.oversampled_img_temp, dims)
    (plan.FFTplan.plan isa Nothing) && (plan.FFTplan.plan = plan_fft!(plan.oversampled_img_temp2, dims))
    plan.FFTplan.plan * plan.oversampled_img_temp2
    fftshift!(plan.oversampled_img_temp, plan.oversampled_img_temp2, dims)
    
    !(plan.progressTitle isa Nothing) && next!(p)

    # Interpolate
    if ndim == all_dims
        shaped_img = reshape(plan.oversampled_img_temp, 1, size(plan.oversampled_img_temp)...)
        shaped_ksp = reshape(ksp, 1, :)
    else
        shaped_img = reshape(plan.oversampled_img_temp, :, size(plan.oversampled_img_temp)[end-ndim+1:end]...)
        batch_size = size(shaped_img, 1)
        shaped_ksp = reshape(ksp, batch_size, :)
    end
    shaped_coord = reshape(plan.coord, ndim, :)
    interpolate!(shaped_ksp, shaped_img, shaped_coord, plan.kernel, plan.width, p)
    ksp
end

function _nufft_backw(img::AbstractArray{Complex{T}}, plan::NUFFT_plan{T}, ksp::AbstractArray{Complex{T}}) where T
    
    ndim = size(plan.coord, 1)
    all_dims = ndims(img)
    
    if !(plan.progressTitle isa Nothing)
        (p = Progress(100, plan.progress_dt, plan.progressTitle))
    else
        p = nothing
    end
    
    # Gridding
    oversampled_shape = size(plan.oversampled_img_temp)
    fill!(plan.oversampled_img_temp, zero(eltype(img))) # initialize oversized buffer
    if ndim == all_dims
        shaped_img = reshape(plan.oversampled_img_temp, 1, size(plan.oversampled_img_temp)...)
        shaped_ksp = reshape(ksp, 1, :)
    else
        shaped_img = reshape(plan.oversampled_img_temp, :, size(plan.oversampled_img_temp)[end-ndim+1:end]...)
        batch_size = size(shaped_img, 1)
        shaped_ksp = reshape(ksp, batch_size, :)
    end
    shaped_coord = reshape(plan.coord, ndim, :)
    gridding!(shaped_img, shaped_ksp, shaped_coord, plan.kernel, plan.width, plan, p)
    
    !(plan.progressTitle isa Nothing) && next!(p)
    
    # IFFT
    dims = tuple((all_dims-ndim+1:all_dims)...)
    ifftshift!(plan.oversampled_img_temp2, plan.oversampled_img_temp, dims)
    (plan.iFFTplan.plan isa Nothing) && (plan.iFFTplan.plan = plan_ifft!(plan.oversampled_img_temp2, dims))
    plan.iFFTplan.plan * plan.oversampled_img_temp2
    fftshift!(plan.oversampled_img_temp, plan.oversampled_img_temp2, dims)
    
    !(plan.progressTitle isa Nothing) && next!(p)

    # "Crop" and scale
    # Note that img_temp is a view to the centered region of oversampled_img_temp, so cropping is done implicitly.
    oshape = size(plan.img_temp)
    scaling_factor = convert(eltype(ksp), prod(oversampled_shape[end-ndim+1:end]) / √(prod(oshape[end-ndim+1:end])))
    img .= plan.img_temp .* scaling_factor
        
    # Apodize
    _apodize!(img, ndim, plan.oversamp, plan.width, plan.β)
    
    !(plan.progressTitle isa Nothing) && finish!(p)
    
    img
end

function LinearAlgebra.mul!(
        output::AbstractArray{Complex{T}},
        plan::NUFFT_plan{T},
        input::AbstractArray{Complex{T}}) where T
    img = plan.adjoint ? output : input
    
    ndim = size(plan.coord, 1)
    @assert(ndims(img) ≥ ndim,
        "The size of coord along the last dimension should be greater or equal then the dimensionality of input.")
    
    @assert(size(plan.img_temp) == size(img),
        "This plan was created for images of size $(size(plan.img_temp)), but image of size $(size(img)) is given.")
        
    is_complex(x::AbstractArray{Complex{T}}) where T = true
    is_complex(x::AbstractArray{T}) where T = false
    get_complex_subtype(x::AbstractArray{Complex{T}}) where T = T
    if is_complex(input)
        @assert(get_complex_subtype(input) == eltype(plan.coord),
            "Precision of eltype of input and coord should match: $(get_complex_subtype(input)) in $(eltype(input)) vs $(eltype(plan.coord))")
    else
        @assert(eltype(input) == eltype(plan.coord),
            "Precision of eltype of input and coord should match: $(eltype(input)) vs $(eltype(plan.coord))")
    end
    if is_complex(output)
        @assert(get_complex_subtype(output) == eltype(plan.coord),
            "Precision of eltype of output and coord should match: $(get_complex_subtype(output)) in $(eltype(output)) vs $(eltype(plan.coord))")
    else
        @assert(eltype(output) == eltype(plan.coord),
            "Precision of eltype of output and coord should match: $(eltype(output)) vs $(eltype(plan.coord))")
    end
    
    plan.adjoint ? _nufft_backw(output, plan, input) : _nufft_forw(output, plan, input)
end

function Base.:*(plan::NUFFT_plan{T}, input::AbstractArray{Complex{T}}) where T
    
    if plan.adjoint
        output = plan.img_temp
    else
        ndim = size(plan.coord, 1)
        batch_shape = size(input)[1:end-ndim]
        pts_shape = size(plan.coord)[2:end]
        output = zeros(eltype(input), (batch_shape..., pts_shape...))
    end
    
    mul!(output, plan, input)
end

function Base.:*(plan::NUFFT_plan{T}, input::AbstractArray{T}) where T
    plan * convert.(Complex{eltype(input)}, input)
end

@doc raw"""
Non-uniform Fast Fourier Transform.

**Arguments**:
- `coord (ArrayType{T})`: Fourier domain coordinate array of shape ``(m_l, \ldots, m_1, ndim)``.
    ``ndim`` determines the number of dimensions to apply the nufft.
    `coord[..., i]` should be scaled to have its range ``[-n_i \div 2, n_i \div 2]``.
- `input (ArrayType{T} or ArrayType{Complex{T}})`: input signal domain array of shape
    ``(n_k, \ldots, n_{ndim + 1}, n_{ndim}, \ldots, n_2, n_1)``,
    where ``ndim`` is specified by `size(coord)[end]`. The nufft
    is applied on the last ``ndim`` axes, and looped over
    the remaining axes. `ArrayType` can be any `AbstractArray`.
- `oversamp (Real)`: oversampling factor (default: $1.25$)
- `width (Int)`: interpolation kernel full-width in terms of
    oversampled grid. (default: $4$)
- `n (Int)`: number of sampling points of the interpolation kernel. (default: $128$)

**Returns**:
- `ArrayType{Complex{T}}`: Fourier domain data of shape
    ``(n_k, \ldots, n_{ndim + 1}, m_l, \ldots, m_1)``.

**References**:
- Fessler, J. A., & Sutton, B. P. (2003).
  "Nonuniform fast Fourier transforms using min-max interpolation",
  *IEEE Transactions on Signal Processing*, 51(2), 560-574.
- Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
  "Rapid gridding reconstruction with a minimal oversampling ratio,"
  *IEEE transactions on medical imaging*, 24(6), 799-808.

"""
function nufft(
        coord::AbstractArray{T},
        input::AbstractArray{Complex{T}};
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T}
    plan = nufft_plan(coord, input, oversamp=oversamp, width=width, n=n, progressTitle=progressTitle, progress_dt=progress_dt)
    plan * input
end

function nufft(
        coord::AbstractArray{T},
        input::AbstractArray{T};
        oversamp::T = 1.25,
        width::Int = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T}
    nufft(coord, convert.(Complex, input), oversamp=oversamp, width=width, n=n, progressTitle=progressTitle, progress_dt=progress_dt)
end

function nufft!(
        output::AbstractArray{Complex{T}},
        coord::AbstractArray{T},
        input::AbstractArray{Complex{T}};
        oversamp::Real = 1.25,
        width::Real = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T}
    plan = nufft_plan(coord, input, oversamp=oversamp, width=width, n=n, progressTitle, progress_dt)
    mul!(output, plan, input)
end

function nufft!(
        output::AbstractArray{Complex{T}},
        coord::AbstractArray{T},
        input::AbstractArray{T};
        oversamp::T = 1.25,
        width::Real = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T<:Real}
    nufft!(output, coord, convert.(Complex, input), oversamp=oversamp, width=width, n=n,
        progressTitle=progressTitle, progress_dt=progress_dt)
end

@doc raw"""
Adjoint non-uniform Fast Fourier Transform.

**Arguments**:
- `coord (ArrayType{T})`: Fourier domain coordinate array of shape ``(m_l, \ldots, m_1, ndim)``.
    ``ndim`` determines the number of dimensions to apply the nufft.
    `coord[..., i]` should be scaled to have its range ``[-n_i \div 2, n_i \div 2]``.
- `input (ArrayType{T} or ArrayType{Complex{T}})`: input Fourier domain array of shape
    ``(n_k, \ldots, n_{l + 1}, m_l, \ldots, m_1)``,
    where ``ndim`` is specified by `size(coord)[end]`.
    That is, the last dimensions
    of input must match the first dimensions of coord.
    The nufft_adjoint is applied on the last coord.ndim - 1 axes,
    and looped over the remaining axes.
- `oshape (NTuple{N, Int})`: output shape of the form
            ``(o_l, \ldots, o_{ndim + 1}, n_{ndim}, \ldots, n_2, n_1)``. (optional)
- `oversamp (Real)`: oversampling factor (default: $1.25$)
- `width (Int)`: interpolation kernel full-width in terms of
    oversampled grid. (default: $4$)
- `n (Int)`: number of sampling points of the interpolation kernel. (default: $128$)

**Returns**:
- `ArrayType{Complex{T}}`: Fourier domain data of shape
    ``(n_k, \ldots, n_{ndim + 1}, n_{ndim}, \ldots, n_2, n_1)`` or
    ``(o_l, \ldots, o_{ndim + 1}, n_{ndim}, \ldots, n_2, n_1)`` if `oshape` is given.

**References**:
- Fessler, J. A., & Sutton, B. P. (2003).
  "Nonuniform fast Fourier transforms using min-max interpolation",
  *IEEE Transactions on Signal Processing*, 51(2), 560-574.
- Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005).
  "Rapid gridding reconstruction with a minimal oversampling ratio,"
  *IEEE transactions on medical imaging*, 24(6), 799-808.

"""
function nufft_adjoint(
        coord::AbstractArray{T},
        input::AbstractArray{Complex{T}};
        oshape::Union{NTuple{N, Int}, Nothing} = nothing,
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T, N}
    (oshape isa Nothing && ndims(input) > 1) && (oshape = tuple(size(input)[1:end-ndims(coord)+1]..., ..))
    plan = nufft_plan(coord, oshape, oversamp=oversamp, width=width, n=n, progressTitle=progressTitle, progress_dt=progress_dt)
    plan' * input
end

function nufft_adjoint(
        coord::AbstractArray{T},
        input::AbstractArray{T};
        oshape::Union{NTuple{N, Int}, Nothing} = nothing,
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T, N}
    nufft_adjoint(coord, convert.(Complex, input), oshape=oshape, oversamp=oversamp, width=width, n=n, progressTitle=progressTitle, progress_dt=progress_dt)
end

function nufft_adjoint!(
        output::AbstractArray{Complex{T}},
        coord::AbstractArray{T},
        input::AbstractArray{Complex{T}};
        oshape::Union{NTuple{N, Int}, Nothing} = nothing,
        oversamp::Real = 1.25,
        width::Real = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T, N}
    (oshape isa Nothing && ndims(input) > 1) && (oshape = tuple(size(input)[1:end-1]..., ..))
    plan = nufft_plan(coord, oshape, oversamp=oversamp, width=width, n=n,
        progressTitle=progressTitle, progress_dt=progress_dt)
    mul!(output, plan', input)
end

function nufft_adjoint!(
        output::AbstractArray{Complex{T}},
        coord::AbstractArray{T},
        input::AbstractArray{T};
        oshape::Union{NTuple{N, Int}, Nothing} = nothing,
        oversamp::Real = 1.25,
        width::Real = 4,
        n::Int = 128,
        progressTitle::Union{String, Nothing} = nothing,
        progress_dt::Real = 1) where {T, N}
    nufft_adjoint!(output, coord, convert.(Complex, input), oshape=oshape, oversamp=oversamp, width=width, n=n,
        progressTitle=progressTitle, progress_dt=progress_dt)
end

"""
Estimate array shape from coordinates.

Shape is estimated by the difference between maximum and minimum of
coordinates along each dimension.

Args:
    `coord (AbstractArray)`: Coordinates.
"""
function estimate_shape(coord)
    dims = tuple((2:ndims(coord))...)
    return floor.(Int, Array(dropdims(maximum(coord, dims=dims), dims=dims) -
        dropdims(minimum(coord, dims=dims), dims=dims)))
end

function fftshift!(
        output::AbstractArray,
        input::AbstractArray,
        dims::NTuple{N,Int}) where {N}
    
    @assert input !== output "input and output must be two distinct arrays"
    @assert any(dims .> 0) "dims can contain only positive values!"
    @assert any(dims .<= ndims(input)) "dims cannot contain larger value than ndims(input) (=$(ndims(input)))"
    @assert size(output) == size(input) "input and output must have the same size"
    @assert eltype(output) == eltype(input) "input and output must have the same eltype"
    
    shifts = [dim in dims ? size(input, dim) ÷ 2 : 0 for dim in 1:ndims(input)]
    circshift!(output, input, shifts)
    
end

function ifftshift!(
        output::AbstractArray,
        input::AbstractArray,
        dims::NTuple{N,Int}) where {N}
    
    @assert input !== output "input and output must be two distinct arrays"
    @assert any(dims .> 0) "dims can contain only positive values!"
    @assert any(dims .<= ndims(input)) "dims cannot contain larger value than ndims(input) (=$(ndims(input)))"
    @assert size(output) == size(input) "input and output must have the same size"
    @assert eltype(output) == eltype(input) "input and output must have the same eltype"
    
    shifts = [dim in dims ? size(input, dim) ÷ 2 + size(input, dim) % 2 : 0 for dim in 1:ndims(input)]
    circshift!(output, input, shifts)
    
end

fftshift!(output::AbstractArray, input::AbstractArray, dims::Int) =
    fftshift!(output, input, (dims,))

ifftshift!(output::AbstractArray, input::AbstractArray, dims::Int) =
    ifftshift!(output, input, (dims,))

"""
FFT function that supports centering.

**Arguments**:
    input (AbstractArray{T,N}): input array.
    dims (NTuple{K,Int}): Axes over which to compute the FFT (optional).

**Returns**:
    AbstractArray: FFT result.

"""
function centering_fft!(
        input::AbstractArray{Complex{T},N},
        dims::Union{NTuple{K,Int},Nothing} = nothing) where {N,K,T}
    (dims isa Nothing) && (dims = tuple(collect(1:ndims(input))...))
    output = ifftshift(input, dims)
    fft!(output, dims)
    fftshift!(input, output, dims)
    input
end

"""
inverse FFT function that supports centering.

**Arguments**:
    input (AbstractArray{T,N}): input array.
    dims (NTuple{K,Int}): Axes over which to compute the inverse FFT (optional).

**Returns**:
    AbstractArray: inverse FFT result.

"""
function centering_ifft!(
        input::AbstractArray{Complex{T},N},
        dims::Union{NTuple{K,Int},Nothing} = nothing) where {N,K,T}
    (dims isa Nothing) && (dims = tuple(collect(1:ndims(input))...))
    output = similar(input)
    ifftshift!(output, input, dims)
    ifft!(output, dims)
    fftshift!(input, output, dims)
    input
end

function crop(input::AbstractArray, oshape::NTuple)

    ishape1, oshape1 = _expand_shapes(size(input), oshape)
    ishift = [max(i ÷ 2 - o ÷ 2, 0) for (i, o) in zip(collect(ishape1), collect(oshape1))]
    oshift = [max(o ÷ 2 - i ÷ 2, 0) for (i, o) in zip(collect(ishape1), collect(oshape1))]

    copy_shape = [min(i - si, o - so)
                  for (i, si, o, so) in zip(collect(ishape1), ishift, collect(oshape1), oshift)]
    islice = collect(si+1:si+c for (si, c) in zip(ishift, copy_shape))
    return input[islice...]
end

function _get_oversamp_shape(shape, ndim, oversamp)
    return tuple(vcat(shape[1:end-ndim]..., [ceil(Int, oversamp * i) for i in shape[end-ndim+1:end]]...)...)
end

function _apodize!(signal::AbstractArray, apodized_dims, oversamp, width, β)
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
        window = recip_iFFT_Kaiser_Bessel_kernel.(0:axis_size-1)
        
        # Apply point-wise along selected axis, broadcast along all other dimensions
        broadcast_shape = ones(Int, all_dims)
        broadcast_shape[axis] = axis_size
        signal .*= reshape(window, broadcast_shape...)
    end
    return signal
end

function _scale_coord(coord, shape, oversamp)
    ndim = size(coord, 1)
    pts_dims_broadcast = fill(1, ndims(coord) - 1)
    scaling = [convert(eltype(coord), ceil(oversamp * d) / d) for (i,d) in enumerate(shape[end-ndim+1:end])]
    scaling = reshape(scaling, ndim, pts_dims_broadcast...)
    shift = [convert(eltype(coord), ceil(oversamp * d) ÷ 2) for (i,d) in enumerate(shape[end-ndim+1:end])]
    shift = reshape(shift, ndim, pts_dims_broadcast...)
    return coord .* scaling .+ shift
end

window_kaiser_bessel(x::Real, m::Int, β::Real) = 1 / m * besseli(0, β * √(1 - x^2))

function _expand_shapes(shapes...)
    max_ndim = maximum(length, shapes)
    return map(shape -> tuple(vcat(repeat([1], max_ndim - length(shape)), collect(shape))...), shapes)
end

@doc raw"""
    Interpolation from array to points specified by coordinates.

    Let ``x`` be the input, ``y`` be the output,
    ``c`` be the coordinates, ``W`` be the kernel width,
    and ``K`` be the interpolation kernel, then the function computes,

    ```math
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]
    ```

    **Arguments**:
        `input (AbstractArray)`: Input array of shape
            ``(n_1, n_2, \ldots, n_{ndim})``.
        `width (Int)`: Interpolation kernel full-width.
        `kernel (AbstractArray{T, 1})`: Interpolation kernel.
        `coord (AbstractArray)`: Coordinate array of shape ``(m_l, \ldots, m_1, ndim)``

    **Returns**:
        `output (AbstractArray)`: Output array of shape ``(m_l, \ldots, m_1)``
    """
function interpolate!(
        output::AbstractArray{Complex{T}, 2},
        input::AbstractArray{Complex{T}, D},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        p::Union{Progress,Nothing} = nothing) where {T, D}
    increment_cycle = size(coord, 2) ÷ 98
    Threads.@threads for point_index in 1:size(coord, 2)
        _interpolate_point!(output, input, coord, kernel, width, point_index)
        !(p isa Nothing) && (point_index % increment_cycle == 0) && next!(p)
    end
end

function interpolate!(
        output::AbstractArray{T, 2},
        input::AbstractArray{T, D},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        p::Union{Progress,Nothing} = nothing) where {T, D}
    increment_cycle = size(coord, 2) ÷ 98
    Threads.@threads for point_index in 1:size(coord, 2)
        _interpolate_point!(output, input, coord, kernel, width, point_index)
        !(p isa Nothing) && (point_index % increment_cycle == 0) && next!(p)
    end
end

@doc raw"""
    Gridding of points specified by coordinates to array.

    Let ``x`` be the input, ``y`` be the output,
    ``c`` be the coordinates, ``W`` be the kernel width,
    and ``K`` be the interpolation kernel, then the function computes,

    ```math
        y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) x[i]
    ```

    **Arguments**:
        `input (AbstractArray)`: Input array of shape
            ``(m_l, \ldots, m_1)``.
        `oshape (Ntuple{N, Int})`: Shape of output
        `width (Int)`: Interpolation kernel full-width.
        `kernel (AbstractArray{T, 1})`: Interpolation kernel.
        `coord (AbstractArray{T})`: Coordinate array of shape ````(m_l, \ldots, m_1, ndim)``

    **Returns**:
        `output (AbstractArray)`: Output array.
    """    
function gridding!(
        output::AbstractArray{Complex{T}, D},
        input::AbstractArray{Complex{T}, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        plan::NUFFT_plan,
        p::Union{Progress,Nothing} = nothing) where {T, D}
    _gridding!(output, input, coord, kernel, width, plan, p)
end

function gridding!(
        output::AbstractArray{T, D},
        input::AbstractArray{T, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        plan::NUFFT_plan,
        p::Union{Progress,Nothing} = nothing) where {T, D}
    _gridding!(output, input, coord, kernel, width, plan, p)
end

function lin_interpolate(kernel::AbstractArray{T,1}, x::T) where T
    x >= 1 && return zero(x)
    
    n = length(kernel)
    idx = floor(Int, x * n)
    frac = x * n - idx

    left = kernel[idx + 1]
    right = idx < n - 1 ? kernel[idx + 2] : zero(x)
    
    return (1 - frac) * left + frac * right
end

macro interpolate_point(ndim)
    batch_plus_ndim = ndim + 1
    esc(quote
        input_shape = size(input)
        batch_size = input_shape[1]
        @nextract($ndim, n, d -> input_shape[$batch_plus_ndim-d+1])
        @nextract($ndim, interval_middle,
                d -> coord[$batch_plus_ndim - d, point_index])
        @nextract($ndim, interval_start,
                d -> ceil(Int, interval_middle_d - width / 2))
        @nextract($ndim, interval_end,
                d -> floor(Int, interval_middle_d + width / 2))
        $(Symbol(:w_, batch_plus_ndim)) = 1
        @nloops($ndim, i, d -> interval_start_d:interval_end_d,
            d -> begin
                pos_d = mod(i_d, n_d) + 1
                w_d = w_{d+1} *
                    lin_interpolate(kernel, convert(eltype(kernel), abs(i_d - interval_middle_d) / (width / 2)))
            end,
            begin
                position = @ntuple($ndim, d -> pos_{$ndim-d+1})
                for b in 1:batch_size
                    @inbounds output[b, point_index] += w_1 * input[b, position...]
                end
            end)
    end)
end

macro gridding_point(ndim)
    batch_plus_ndim = ndim + 1
    esc(quote
        output_shape = size(output)
        batch_size = output_shape[1]
        @nextract($ndim, n, d -> output_shape[$batch_plus_ndim-d+1])
        @nextract($ndim, interval_middle,
                d -> coord[$batch_plus_ndim - d, point_index])
        @nextract($ndim, interval_start,
                d -> ceil(Int, interval_middle_d - width / 2))
        @nextract($ndim, interval_end,
                d -> floor(Int, interval_middle_d + width / 2))
        $(Symbol(:w_, batch_plus_ndim)) = 1
        @nloops($ndim, i, d -> interval_start_d:interval_end_d,
            d -> begin
                pos_d = mod(i_d, n_d) + 1
                w_d = w_{d+1} *
                    lin_interpolate(kernel, convert(eltype(kernel), abs(i_d - interval_middle_d) / (width / 2)))
            end,
            begin
                position = @ntuple($ndim, d -> pos_{$ndim-d+1})
                for b in 1:batch_size
                    @inbounds output[b, position...] += w_1 * input[b, point_index] 
                end
            end)
    end)
end

# It is not a necessary function, but might help to debug transformed code
# Add line numbers matching the lines of transformed code (the call stack will be more 
# informative then if an error occures inside the transformed code)
function addLineNumbers(expr)
    # Remove previous line numbers and add correct ones instead
    result = Meta.parse(string(MacroTools.striplines(expr)))
    # fix LineNumberNodes to be more informative
    MacroTools.postwalk(result) do x
        x isa LineNumberNode ?
            LineNumberNode(x.line, Symbol("generated_code_of_interpolate")) :
            x
    end
end

function pretty_print_expression(expr; withLineNumbers=false)
    expr = MacroTools.prewalk(x -> MacroTools.isgensym(x) ? Symbol(MacroTools.gensymname(x)) : x, expr) 
    expr = MacroTools.prewalk(unblock, expr)
    expr = addLineNumbers(expr)
    print(withLineNumbers ? expr : MacroTools.striplines(expr))
end

# This way you can see the generated code for _interpolate_point! and _gridding_point! generated functions
#pretty_print_expression(@macroexpand(@gridding_point(3)), withLineNumbers=false)

@generated function _interpolate_point!(
        output::AbstractArray{Complex{T}, 2},
        input::AbstractArray{Complex{T}, D},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        point_index::Int) where {T, D}
    quote
        @interpolate_point $(D-1)
    end
end

@generated function _interpolate_point!(
        output::AbstractArray{T, 2},
        input::AbstractArray{T, D},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        point_index::Int) where {T, D}
    quote
        @interpolate_point $(D-1)
    end
end

@generated function _gridding_point!(
        output::AbstractArray{Complex{T}, D},
        input::AbstractArray{Complex{T}, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        point_index::Int) where {T, D, N}
    quote
        @gridding_point $(D-1)
    end
end

@generated function _gridding_point!(
        output::AbstractArray{T, D},
        input::AbstractArray{T, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        point_index::Int) where {T, D, N}
    quote
        @gridding_point $(D-1)
    end
end

function _gridding!(
        output::AbstractArray{T1, D},
        input::AbstractArray{T1, 2},
        coord::AbstractArray{T2, 2},
        kernel::AbstractArray{T2, 1},
        width::Int,
        plan::NUFFT_plan,
        p::Union{Progress,Nothing} = nothing) where {T1, T2, D}
    if Threads.nthreads() > 1 && # if threading enabled
            size(input, 2) > 10000 && # if the problem large enough
            Sys.free_memory() * .8 > sizeof(output) * (Threads.nthreads()-1) # if we have enough memory
        #(plan.griddingBuffer.buffers isa Nothing) &&
        #    (plan.griddingBuffer.buffers = [copy(output) for _ in 2:Threads.nthreads()])
        #threaded_output = tuple(output, plan.griddingBuffer.buffers...)
        threaded_output = tuple(output, [copy(output) for _ in 2:Threads.nthreads()]...)
        Threads.@threads for point_index in 1:size(coord, 2)
            _gridding_point!(threaded_output[Threads.threadid()], input, coord, kernel, width, point_index)
            !(p isa Nothing) && next!(p)
        end
        for i in 2:length(threaded_output)
            output .+= threaded_output[i]
        end
    else
        for point_index in 1:size(coord, 2)
            _gridding_point!(output, input, coord, kernel, width, point_index)
            !(p isa Nothing) && next!(p)
        end
    end
    output
end

"""
Resize with zero-padding or cropping.

**Arguments**:
    `input (AbstractArray{T,N})`: Input array.
    `oshape (NTuple{N,Int})`: Output shape.
    `ishift (NTuple{N,Int})`: Input shift (optional).
    `oshift (NTuple{N,Int})`: Output shift (optional).

**Returns**:
    `array`: Zero-padded or cropped result.
"""
function resize(
        input::AbstractArray{T,N},
        oshape::NTuple{N,Int},
        ishift::Union{NTuple{N,Int}, Nothing} = nothing,
        oshift::Union{NTuple{N,Int}, Nothing} = nothing) where {N,T}

    ishape1, oshape1 = _expand_shapes(size(input), oshape)

    if ishape1 == oshape1
        return reshape(input, oshape)
    end

    if ishift isa Nothing
        ishift = [max(i ÷ 2 - o ÷ 2, 0) for (i, o) in zip(collect(ishape1), collect(oshape1))]
    end

    if oshift isa Nothing
        oshift = [max(o ÷ 2 - i ÷ 2, 0) for (i, o) in zip(collect(ishape1), collect(oshape1))]
    end

    copy_shape = [min(i - si, o - so)
                  for (i, si, o, so) in zip(collect(ishape1), ishift, collect(oshape1), oshift)]
    islice = collect(si+1:si+c for (si, c) in zip(ishift, copy_shape))
    oslice = collect(so+1:so+c for (so, c) in zip(oshift, copy_shape))

    output = zeros(eltype(input), oshape1)
    input = reshape(input, ishape1)
    output[oslice...] = input[islice...]

    return reshape(output, oshape)
end

end # end module
