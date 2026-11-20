function batch_score_value(x, p)
    value = zero(eltype(x))
    for i in eachindex(x)
        z = x[i] - p.centre[i]
        value -= z^2 / 2 + z^4 / 16
    end
    return value
end

function batch_score_scalar!(g, x, p)
    for i in eachindex(x)
        z = x[i] - p.centre[i]
        g[i] = -z - z^3 / 4
    end
    return nothing
end

function batch_score!(values, gradients, samples, p)
    z = samples .- p.centre
    values .= vec(sum(.-z .^ 2 ./ 2 .- z .^ 4 ./ 16; dims = 1))
    gradients .= .-z .- z .^ 3 ./ 4
    return nothing
end

function named_batch_score_value(theta, p)
    value = -theta.scale^2 / 2 - theta.offset^2 / 2
    for i in eachindex(theta.weights)
        value += p.powers[i] * log(theta.weights[i])
    end
    return value
end

function named_batch_score_scalar!(g, theta, p)
    for i in eachindex(theta.weights)
        g.weights[i] = p.powers[i] / theta.weights[i]
    end
    g.scale[] = -theta.scale
    g.offset[] = -theta.offset
    return nothing
end

function named_batch_score!(values, gradients, samples, p)
    values .=
        vec(sum(p.powers .* log.(samples.weights); dims = 1)) .- samples.scale .^ 2 ./ 2 .-
        samples.offset .^ 2 ./ 2
    gradients.weights .= p.powers ./ samples.weights
    gradients.scale .= .-samples.scale
    gradients.offset .= .-samples.offset
    return nothing
end
