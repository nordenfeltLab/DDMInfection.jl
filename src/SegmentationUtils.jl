distfeat_transform(img) = ImageMorphology.distance_transform(ImageMorphology.feature_transform(img))
function snr_binarize(img; sigma = 4,kwargs...)
    m,s = sigma_clipped_stats(img)
    
    img .> m + s*sigma
end

otsu_close_seg(img) = closing(img .> otsu_threshold(img))

segmentation_lib = Dict(
    "sigma_clipped" => (x;kwargs...) -> snr_binarize(x;kwargs...),
    "otsu_close" => (x;kwargs...) -> otsu_close_seg(x)
    )

function regionprop_analysis(img;method="sigma_clipped",minsize=150, maxsize=2000, kwargs...)
    seg = segmentation_lib[method](img;kwargs...) |>
        label_components
    seg = sparse(seg)
    counts = countmap(nonzeros(seg))
    for (i, j, v) in zip(findnz(seg)...)
        if counts[v] < minsize || counts[v] > maxsize
            seg[i,j] = 0
        end
    end
    dropzeros!(seg)
    props = ((;r...) for r in regionprops(img, seg; selected=unique(nonzeros(seg))))
    seg, props
end


±(x,n) = (x-n, x+n)
function box_crop(x,y, img; spacing = 50)
    ymax,xmax = size(img)
    yrange = clamp.(y ± spacing, 1, ymax) |> x -> UnitRange(x...)
    xrange = clamp.(x ± spacing, 1, xmax) |> x -> UnitRange(x...)
    
    img[yrange, xrange]
end

function overlap_distance(parent_crop, child_crop)
    
    # TODO: This needs some testing for sparse
    distance_map = distfeat_transform(Array(parent_crop)) .* Array(child_crop .> 0)
    distance = minimum(distance_map[child_crop .> 0])
    indices = findall(x -> x == distance, distance_map)
    indice_vec = child_crop[indices]
    child_labels = countmap(indice_vec[indice_vec .!= 0])
    label = reduce((x, y) -> child_labels[x] ≥ child_labels[y] ? x : y, keys(child_labels))
        
    return (distance = distance, child_label = label)
end