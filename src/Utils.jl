# Utilities
struct SystemInfo
    num_agents::Int
    num_x::Int
    num_us::AbstractArray{Int}
    num_v::Int
end
SystemInfo(num_agents, num_x, num_us) = SystemInfo(num_agents, num_x, num_us, 0)

export SystemInfo
