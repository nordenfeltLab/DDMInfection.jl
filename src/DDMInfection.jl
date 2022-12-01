module DDMInfection

using Images
using SparseArrays
using BlockArrays
using Distances
using LinearAlgebra
using Hungarian
using NearestNeighbors
using ImageMorphology
using ImageSegmentation
using SegmentationUtils
using StatsBase
using RegionProps
using DataFrames
using BioformatsLoader
using DDMFramework
using Random
using JSON

export InfectionState

# Remove CombinedParsers#master from deps
# and change DDMFramework from dev

struct FOV
    x::Float64
    y::Float64
    index::Int64
    #data::Array{DaraFrame}
end

struct Objects
    data::Vector{DataFrame}
end

mutable struct TrackedObjects
    objects::Vector{Objects}
    tracks::DataFrame
end

mutable struct InfectionState
    fovs::Vector{FOV}
    objects::Dict{String, Union{TrackedObjects,Vector{Objects}}}
    config::Dict{String, Any}
end

TrackedObjects() = TrackedObjects(
    Objects[],
    DataFrame(
        label_id = Int[],
        track_id = Int[],
        t = Int[],
        fov_id = Int[]
    )
)

function create_objects(seg_params)
    objects = Dict{String, Union{TrackedObjects, Vector{Objects}}}()
    foreach(seg_params) do (k,v)
        if !haskey(v, "tracking") || !v["tracking"]
            objects[k] = Objects[]
        else
            objects[k] = TrackedObjects()
        end
    end
    objects
end


function InfectionState(params::Dict{String, Any})
    InfectionState(
        FOV[],
        create_objects(params["segmentation"]),
        params
        )
end


include("SegmentationUtils.jl")
include("TrackUtils.jl")


Base.push!(x::Objects, df::DataFrame) = push!(x.data, df)
Base.push!(x::TrackedObjects, data::Objects) = push!(x.objects, data)

function find_fov_id(x::Float64, y::Float64, fov_arr::Vector{FOV})
    findfirst(fov ->
        isapprox(x, fov.x, atol=1.2) &&
        isapprox(y, fov.y, atol=1.2),
        fov_arr
        )
end

function get_fov_id!(fov_arr::Vector{FOV}, x::Float64, y::Float64)
    if isempty(fov_arr)
        push!(fov_arr, FOV(x,y,1))
        1
    else
        fov_id = find_fov_id(x,y, fov_arr)
        
        if isnothing(fov_id)
            max_id = map(x -> x.index, fov_arr) |> maximum
            push!(state, FOV(x,y,max_id+1))
            return max_id+1
        end
        fov_id
    end
end

function get_fov_id!(state::InfectionState, params::Dict{String, Any})
    x = params["stage_pos_x"][end]
    y = params["stage_pos_y"][end]
    id = get_fov_id!(state.fovs, x, y)
end

function push_new_lazy!(fun,d,k)
    if !haskey(d,k)
        push!(d,k => fun())
    end
    return d
end

function parse_image_meta!(state, data)
    push_new_lazy!(state.config, "image_meta") do
        Dict{String, Any}(
            "stage_pos_x" => [],
            "stage_pos_y" => []
        )
    end

    let (mdata, params) = (data["image"].Pixels, state.config["image_meta"])

        push!(params["stage_pos_x"], mdata[:Plane][1][:PositionX])
        push!(params["stage_pos_y"], mdata[:Plane][1][:PositionY])

        push_new_lazy!(params, "img_size") do
            (y = mdata[:SizeY], x = mdata[:SizeX], z = mdata[:SizeZ])
        end
        push_new_lazy!(params, "pixelmicrons") do
            mdata[:PhysicalSizeX]
        end
    end
end

function update_objects!(new_objs, index, state::Vector{Objects})
    if isempty(state)
        push!(state, Objects([new_objs]))
    else
        push!(state[index], new_objs)
    end
end

function update_objects!(new_objs, index, state::TrackedObjects)
    update_objects!(new_objs, index, state.objects)
    n_timepoints = length(state.objects[index].data)
    display(n_timepoints ≥ 2)
    if n_timepoints ≥ 2
        
        update_tracks!(state.tracks, state.objects, index)
    end
    state
end

function get_channel(img, channel_defs::Vector{Dict{String, Any}}, channel_name::String)
    ind = find_channel(channel_defs, channel_name)
    img[ind, :,:]
end

function find_channel(channel_defs, channel_name)
    index = findfirst(x -> x["name"] == channel_name, channel_defs)
    return channel_defs[index]["index"]
end

function drop_empty_dims(img::ImageMeta)
    dims = Tuple(findall(x -> x.val.stop == 1, img.data.axes))
    dropdims(img.data.data, dims=dims)
end

function DDMFramework.handle_update(state::InfectionState, data)
    parse_image_meta!(state, data)
    update_state!(
        state,
        data["image"]
        #drop_empty_dims(data["image"])
    )
end

to_named_tuple(dict::Dict{K,V}) where {K,V} = NamedTuple{Tuple(Iterators.map(Symbol,keys(dict))), NTuple{length(dict),V}}(values(dict))

function update_state!(image, state::InfectionState)
    
    fov_id = get_fov_id!(state, state.config["image_meta"])
    
    labeled_images = Dict{String, Any}()
    
    for (k, seg_p) in state.config["segmentation"]
        img = get_channel(
            image,
            state.config["channel_definitions"],
            k
        )
        lb_image, new_objects = regionprop_analysis(
            img;
            DDMFramework.to_named_tuple(seg_p)...
        )
        
        labeled_images[k] = lb_image
        update_objects!(DataFrame(new_objects), fov_id, state.objects[k])
    end
    
    let params = state.config["analysis"]
        parent = params["parent"]
        child = params["child"]
        all_primary_objects = state.objects[parent].objects[fov_id].data
        latest_objects = all_primary_objects[end]
        append_object_distances!(
            latest_objects,
            labeled_images[parent],
            labeled_images[child]
        )
    end
    
    return "0", state
end


function append_object_distances!(df_primary, primary_lb, secondary_lb; window_halfwidth = 50)
    if !isempty(df_primary)
        stats = map(eachrow(df_primary)) do row
            x,y = round.(Int, (row.centroid_x,row.centroid_y))
            secondary_crop = box_crop(x,y,secondary_lb)

            stats = if sum(secondary_crop[:]) > 0
                primary_crop = box_crop(x, y, primary_lb) .== row.label_id
                overlap_distance(primary_crop, secondary_crop)
            else
                (distance=Inf, child_label=0)
            end
            merge(stats, (;label_id = row.label_id))
        end |> DataFrame

        leftjoin!(df_primary, stats, on=:label_id)
    else
        nothing
    end
end

table_filters = Dict(
     "bin" => function bin(data, sel, n)
         data = filter(!isnan, data)
         if sel == 1
             hi = quantile(data, sel/n)
             row -> row < hi
         elseif sel == n
             lo = quantile(data, (sel-1)/n)
             row -> row > lo
         else
             lo, hi = quantile(data, [(sel-1)/n, sel/n])
             row -> lo <= row < hi
         end
    end,
    ">" => function gt(data,v)
        >(v)
    end,
    "max" => function _max(data)
        ==(maximum(data))
    end,
    "<" => function lt(data,v)
        <(v)
    end,
    "eq" => function eq(data,v)
        ==(v)
    end
)

to_stage_pos(xv,yv,stage_x,stage_y, p) = to_stage_pos(xv,
                                                      yv,
                                                      stage_x,
                                                      stage_y,
                                                      p["system"]["camera_M"],
                                                      p["system"]["pixelmicrons"],
                                                      p["image_meta"]["img_size"].y,
                                                      p["image_meta"]["img_size"].x
                                                  )
function to_stage_pos(xv,yv,stage_x,stage_y, camera_m, pixelmicrons, height, width)
    translation(x, p) =  x .* p.pixelmicrons .+ [p.y p.x]
    centre_coords(x, h, w) = ((x .- ([h w] ./2)) .* [-1 1])'
    camera_M = [camera_m["a11"] camera_m["a12"]; camera_m["a21"] camera_m["a22"]]
    p = (pixelmicrons=pixelmicrons, y=stage_y, x=stage_x, h=height, w=width)
    
    corr_coords = camera_M * centre_coords(hcat(yv,xv), p.h, p.w)
    translation(corr_coords', p)[:]
end

function sample_df(df, n::Int64, seed::Int64 = 1234)
    sel = df.selection
    n_tot = length(sel)
    index = randperm(MersenneTwister(seed), n_tot)[1:min(n, n_tot)]
    
    select(df, sel[index])
end

function collect_objects(objects,fov_arr)
    out = DataFrame()
    for (index, obj) in enumerate(objects)
        fov = fov_arr[index]
        for (t, prop) in enumerate(obj.data)
            df = copy(prop)
            df[!, :fov_id] .= index
            df[!, :t] .= t
            df[!, :stage_x] .= fov.x
            df[!, :stage_y] .= fov.y
            append!(out, df)
        end
    end
    out
end

function collect_objects(objects::TrackedObjects, fov_arr)
    df = collect_objects(objects.objects,fov_arr)
    innerjoin(objects.tracks, df, on = [:fov_id, :t, :label_id])
end

function collect_state(state::InfectionState, args)
    args = Dict(args)
        
    primary = state.config["analysis"]["parent"]
    df = collect_objects(state.objects[primary],state.fovs)
    index = [:fov_id, :track_id]

    @time data = map(gdf for gdf in groupby(df, index)) do gdf
        (gdf=gdf, centroid_x=diff(gdf.centroid_x), centroid_y=diff(gdf.centroid_y))
    end |> DataFrame
    
    stage_pos_transform(x, y, gdf) = to_stage_pos(x,y, gdf.stage_x[1], gdf.stage_y[1], state.config)
    stage_getter(gdf) = [gdf.stage_x[end], gdf.stage_y[end]]
    data = DDMFramework.LazyDF(
        data;
        track = [:gdf] => ByRow(gdf -> Dict("x" => gdf.centroid_x, "y" => gdf.centroid_y)),
        last_x= [:gdf] => ByRow(gdf -> gdf.centroid_x[end]),
        last_y= [:gdf] => ByRow(gdf -> gdf.centroid_y[end]),
        last_stage_pos = [:last_x, :last_y, :gdf] => ByRow(stage_pos_transform),
        fov_id = [:gdf] => ByRow(gdf -> gdf.fov_id[end]),
        track_id = [:gdf] => ByRow(gdf -> gdf.track_id[end]),
        child_distance = [:gdf] => ByRow(gdf -> gdf.distance[end]),
        track_length = [:gdf] => ByRow(gdf -> nrow(gdf)),
        timepoint = [:gdf] => ByRow(gdf -> gdf.t[end]),
        
        stage_pos = [:gdf] => ByRow(gdf -> stage_getter(gdf))
    )
    
    data = if haskey(args, "filter")
        filters = mapfoldl(vcat, args["filter"]; init=Pair{String, Base.Callable}[]) do (column, filters)
            map(filters) do filt
                op = table_filters[filt["op"]](data[!,column], filt["args"]...)
                column => op
            end
        end
        reduce((d, f) -> filter(f, d), filters; init=data)
    else
        data
    end
    
    if haskey(args, "order")
        columns = [o == "asc" ? data[c] : -data[c] for (o, c) in args["order"]]
        key_type = Tuple{eltype.(columns)...}
        by(i) = key_type((c[i] for c in columns))
        data = select(data, sort(1:nrow(data); by))
    end
        
    if haskey(args, "sample")
        n = args["sample"]["n"]
        seed = args["sample"]["seed"]
        sample_df(data, n, seed)
    elseif haskey(args, "limit")
        n = args["limit"]
        DDMFramework.limit(data, n)
    else
        data
    end
end


schema = Dict(
    "query" => "Query",
    "Query" => Dict(
        "infection" => "Infection"
    ),
    "Infection" => Dict(
        "centroid_x" => "Column",
        "centroid_y" => "Column",
        "last_x" => "Column",
        "last_y" => "Column",
        "last_stage_pos" => "Column",
        "track" => "Column",
        "fov_id" => "fov_id",
        "track_id" => "track_id",
        "timepoint" => "timepoint",
        "timepoints" => "timepoint",
        "track_length" => "track_length",
        "stage_pos" => "Column",
        "child_distance" => "child_distance"
        
    ),
    "Column" => Dict(
        "name" => "String"
    )
    
)

resolvers(state) = Dict(
    "Query" => Dict(
        "infection" => (parent, args) -> collect_state(state, args)
    )
)

function DDMFramework.query_state(state::InfectionState, query)
    execute_query(query["query"], schema, resolvers(state)) |> JSON.json
end

function Base.show(io::IO, mime::MIME"text/html", state::InfectionState)
    show(io, mime, collect_objects(state))
end

function readnd2(io)
    mktempdir() do d
        path = joinpath(d, "file.nd2")
        open(path, "w") do iow
            write(iow, read(io))
        end
        BioformatsLoader.bf_import(path)[1]
    end
end

function __init__()
    DDMFramework.register_mime_type("image/nd2", readnd2)
    DDMFramework.add_plugin("infection", InfectionState)
end


export handle_update


end # module