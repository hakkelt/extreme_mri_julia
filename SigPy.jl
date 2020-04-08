module SigPy

using Documenter, EllipsisNotation, FFTW, PaddedViews, SpecialFunctions, Base.Cartesian, Base.Threads

export nufft, nufft_adjoint

@doc raw"""
Non-uniform Fast Fourier Transform.

**Arguments**:
- `input (ArrayType{T} or ArrayType{Complex{T}})`: input signal domain array of shape
    ``(n_k, \ldots, n_{ndim + 1}, n_{ndim}, \ldots, n_2, n_1)``,
    where ``ndim`` is specified by `size(coord)[end]`. The nufft
    is applied on the last ``ndim`` axes, and looped over
    the remaining axes. `ArrayType` can be any `AbstractArray`.
- `coord (ArrayType{T})`: Fourier domain coordinate array of shape ``(m_l, \ldots, m_1, ndim)``.
    ``ndim`` determines the number of dimensions to apply the nufft.
    `coord[..., i]` should be scaled to have its range ``[-n_i \div 2, n_i \div 2]``.
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
        input::AbstractArray{Complex{T}},
        coord::AbstractArray{T},
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128) where {T}
    
    ndim = size(coord)[end]
    @assert(ndims(input) ≥ ndim,
        "The size of coord along the last dimension should be greater or equal then the dimensionality of input.")
    
    β = convert(eltype(coord), π * √(((width / oversamp) * (oversamp - 0.5))^2 - 0.8))
    shape = size(input)
    oversampled_shape = _get_oversamp_shape(shape, ndim, oversamp)

    output = copy(input)

    # Apodize
    _apodize!(output, ndim, oversamp, width, β)

    # Zero-pad
    output /= convert(eltype(input), √(prod(shape[end-ndim+1:end])))
    shift = oversampled_shape .÷ 2 .- shape .÷ 2 .+ 1
    output = PaddedView(0, output, oversampled_shape, shift)

    # FFT
    all_dims = ndims(input)
    output = centering_fft!(convert.(Complex, output), tuple((all_dims-ndim+1:all_dims)...))

    # Interpolate
    coord = _scale_coord(coord, size(input), oversamp)
    x = range(0, stop=n-1, step=1) ./ n
    kernel = convert(eltype(coord), window_kaiser_bessel.(x, width, β))
    return interpolate(output, width, kernel, coord)
    
end

function nufft(
        input::AbstractArray{T},
        coord::AbstractArray{T},
        oversamp::T = 1.25,
        width::Int = 4,
        n::Int = 128) where {T<:Real}
    return nufft(convert.(Complex, img), coord, oversamp, width)
end

@doc raw"""
Adjoint non-uniform Fast Fourier Transform.

**Arguments**:
- `input (ArrayType{T} or ArrayType{Complex{T}})`: input Fourier domain array of shape
    ``(n_k, \ldots, n_{l + 1}, m_l, \ldots, m_1)``,
    where ``ndim`` is specified by `size(coord)[end]`.
    That is, the last dimensions
    of input must match the first dimensions of coord.
    The nufft_adjoint is applied on the last coord.ndim - 1 axes,
    and looped over the remaining axes.
- `coord (ArrayType{T})`: Fourier domain coordinate array of shape ``(m_l, \ldots, m_1, ndim)``.
    ``ndim`` determines the number of dimensions to apply the nufft.
    `coord[..., i]` should be scaled to have its range ``[-n_i \div 2, n_i \div 2]``.
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
        input::AbstractArray{Complex{T}},
        coord::AbstractArray{T},
        oshape::Union{NTuple{N, Int}, Nothing} = nothing,
        oversamp::Real = 1.25,
        width::Int = 4,
        n::Int = 128) where {T, N}
    
    ndim = size(coord)[end]
    
    β = convert(eltype(coord), π * √(((width / oversamp) * (oversamp - 0.5))^2 - 0.8))
    (oshape isa Nothing) && (oshape = tuple(size(input)[1:end-ndims(coord)+1]..., estimate_shape(coord)...))
    oversampled_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    x = range(0, stop=n-1, step=1) ./ n
    kernel = convert.(eltype(coord), window_kaiser_bessel.(x, width, β))
    output = gridding(input, oversampled_shape, width, kernel, coord)

    # IFFT
    all_dims = ndims(output)
    output = centering_ifft!(convert.(Complex, output), tuple((all_dims-ndim+1:all_dims)...))

    # Crop
    output = resize(output, oshape)
    output .*= convert.(eltype(input), prod(oversampled_shape[end-ndim+1:end]) ./ √(prod(oshape[end-ndim+1:end])))

    # Apodize
    _apodize!(output, ndim, oversamp, width, β)

    return output
end

"""
Estimate array shape from coordinates.

Shape is estimated by the different between maximum and minimum of
coordinates in each axis.

Args:
    `coord (AbstractArray)`: Coordinates.
"""
function estimate_shape(coord)
    dims = tuple((1:ndims(coord)-1)...)
    return floor.(Int, dropdims(maximum(coord, dims=dims), dims=dims) -
        dropdims(minimum(coord, dims=dims), dims=(dims)))
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
    output = ifftshift(input, dims)
    ifft!(output, dims)
    fftshift!(input, output, dims)
    input
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

function _get_oversamp_shape(shape, ndim, oversamp)
    return tuple(vcat(shape[1:end-ndim]..., [ceil(Int, oversamp * i) for i in shape[end-ndim+1:end]]...)...)
end

function _apodize!(signal, apodized_dims, oversamp, width, β)
    
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
    ndim = size(coord)[end]
    broadcast_dims = tuple(repeat([1], ndims(coord)-1)..., ndim)
    scale = convert.(eltype(coord), reshape([ceil(oversamp * i) / i for i in shape[end-ndim+1:end]], broadcast_dims))
    shift = convert.(eltype(coord), reshape([ceil(oversamp * i) ÷ 2 for i in shape[end-ndim+1:end]], broadcast_dims))
    return scale .* coord .+ shift
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
function interpolate(
        input::AbstractArray,
        width::Int,
        kernel::AbstractArray{T, 1},
        coord::AbstractArray{T}) where {T<:Real}
    ndim = size(coord, 2)
    npts = size(coord, 1)
    @assert(ndims(input) ≥ ndim, "The size of coord along the last dimension
        should be greater or equal then the dimensionality of input.")
    
    is_complex(x::AbstractArray{Complex{T}}) where T = true
    is_complex(x::AbstractArray{T}) where T = false
    get_complex_subtype(x::AbstractArray{Complex{T}}) where T = T
    if is_complex(input)
        @assert(get_complex_subtype(input) == eltype(coord),
            "Precision of eltype of input and coord should match: $(get_complex_subtype(input)) in $(eltype(input)) vs $(eltype(coord))")
    else
        @assert(eltype(input) == eltype(coord),
            "Precision of eltype of input and coord should match: $(eltype(input)) vs $(eltype(coord))")
    end
    
    batch_shape = size(input)[1:end-ndim]
    batch_size = prod(batch_shape)

    pts_shape = size(coord)[1:end-1]
    npts = prod(pts_shape)

    input = reshape(input, tuple(batch_size, size(input)[end-ndim+1:end]...))
    coord = reshape(coord, (npts, ndim))
    output = zeros(eltype(input), (batch_size, npts))

    _interpolate!(output, input, coord, kernel, width)

    return reshape(output, tuple(batch_shape..., pts_shape...))
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
function gridding(
        input::AbstractArray,
        oshape::NTuple{D, Int},
        width::Int,
        kernel::AbstractArray{T, 1},
        coord::AbstractArray{T}) where {T<:Real, D}
    
    ndim = size(coord)[end]
    batch_shape = oshape[1:end-ndim]
    batch_size = prod(batch_shape)
    
    pts_shape = size(coord)[1:end-1]
    npts = prod(pts_shape)
    
    is_complex(x::AbstractArray{Complex{T}}) where T = true
    is_complex(x::AbstractArray{T}) where T = false
    get_complex_subtype(x::AbstractArray{Complex{T}}) where T = T
    if is_complex(input)
        @assert(get_complex_subtype(input) == eltype(coord),
            "Precision of eltype of input and coord should match: $(get_complex_subtype(input)) in $(eltype(input)) vs $(eltype(coord))")
    else
        @assert(eltype(input) == eltype(coord),
            "Precision of eltype of input and coord should match: $(eltype(input)) vs $(eltype(coord))")
    end

    input = reshape(input, tuple(batch_size, npts))
    coord = reshape(coord, (npts, ndim))
    output = zeros(eltype(input), tuple(batch_size, oshape[end-ndim+1:end]...))

    _gridding!(output, input, coord, kernel, width)

    return reshape(output, oshape)
end

function lin_interpolate(kernel::AbstractArray{T,1}, x::T)::T where T
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
                d -> coord[point_index, $batch_plus_ndim - d])
        @nextract($ndim, interval_start,
                d -> ceil(Int, interval_middle_d - width / 2))
        @nextract($ndim, interval_end,
                d -> floor(Int, interval_middle_d + width / 2))
        $(Symbol(:w_, batch_plus_ndim)) = 1
        @nloops($ndim, i, d -> interval_start_d:interval_end_d,
            d -> w_d = w_{d+1} *
                lin_interpolate(kernel, convert(eltype(kernel), abs(i_d - interval_middle_d) / (width / 2))),
            for b in 1:batch_size
                output[b, point_index] += w_1 * input[b,
                    @ntuple($ndim, d -> (i_{$ndim-d+1} + n_{$ndim-d+1}) % n_{$ndim-d+1} + 1)...]
            end)
    end)
end

macro gridding_point(ndim)
    batch_plus_ndim = ndim + 1
    esc(quote
        output_shape = size(output[1])
        batch_size = output_shape[1]
        @nextract($ndim, n, d -> output_shape[$batch_plus_ndim-d+1])
        @nextract($ndim, interval_middle,
                d -> coord[point_index, $batch_plus_ndim - d])
        @nextract($ndim, interval_start,
                d -> ceil(Int, interval_middle_d - width / 2))
        @nextract($ndim, interval_end,
                d -> floor(Int, interval_middle_d + width / 2))
        $(Symbol(:w_, batch_plus_ndim)) = 1
        @nloops($ndim, i, d -> interval_start_d:interval_end_d,
            d -> w_d = w_{d+1} *
                lin_interpolate(kernel, convert(eltype(kernel), abs(i_d - interval_middle_d) / (width / 2))),
            for b in 1:batch_size
                output[thread_id][b, @ntuple($ndim, d -> (i_{$ndim-d+1} + n_{$ndim-d+1}) % n_{$ndim-d+1} + 1)...] +=
                    w_1 * input[b, point_index] 
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

function _interpolate!(
        output::AbstractArray{Complex{T}, 2},
        input::AbstractArray{Complex{T}, D},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    Threads.@threads for point_index in 1:size(coord, 1)
        _interpolate_point!(output, input, coord, kernel, width, point_index)
    end
end

function _interpolate!(
        output::AbstractArray{T, 2},
        input::AbstractArray{T, D},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    Threads.@threads for point_index in 1:size(coord, 1)
        _interpolate_point!(output, input, coord, kernel, width, point_index)
    end
end

@generated function _gridding_point!(
        output::NTuple{N, AbstractArray{Complex{T}, D}},
        input::AbstractArray{Complex{T}, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        point_index::Int,
        thread_id::Int) where {T, D, N}
    quote
        @gridding_point $(D-1)
    end
end

@generated function _gridding_point!(
        output::NTuple{N, AbstractArray{T, D}},
        input::AbstractArray{T, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int,
        point_index::Int,
        thread_id::Int) where {T, D, N}
    quote
        @gridding_point $(D-1)
    end
end

function __gridding!(
        output::AbstractArray,
        input::AbstractArray{Complex{T}, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    if Threads.nthreads() > 1 && # if threading enabled
            size(input, 2) > 10000 && # if the problem large enough
            Sys.free_memory() * .8 > sizeof(output) * (Threads.nthreads()-1) # if we have enough memory
        threaded_output = tuple(output, [copy(output) for _ in 2:Threads.nthreads()]...)
        Threads.@threads for point_index in 1:size(coord, 1)
            _gridding_point!(threaded_output, input, coord, kernel, width, point_index, Threads.threadid())
        end
        for i in 2:length(threaded_output)
            output .+= threaded_output[i]
        end
    else
        for point_index in 1:size(coord, 1)
            _gridding_point!((output,), input, coord, kernel, width, point_index, 1)
        end
    end
end

function _gridding!(
        output::AbstractArray{Complex{T}, D},
        input::AbstractArray{Complex{T}, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    __gridding!(output, input, coord, kernel, width)
end

function _gridding!(
        output::AbstractArray{T, D},
        input::AbstractArray{T, 2},
        coord::AbstractArray{T, 2},
        kernel::AbstractArray{T, 1},
        width::Int) where {T, D}
    __gridding!(output, input, coord, kernel, width)
end

end # end module
