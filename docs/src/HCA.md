# Hierarchical Cluster Analysis (HCA) 

## Introduction

The [**HCA**](https://en.wikipedia.org/wiki/Hierarchical_clustering) is an unsupervised clustering approach mainly based on the distances of the measurements from each other. It is an agglomerative approach, thus starting with each individual measurement as a cluster and then grouping them to build a final cluster that includes all the measurements.   

## How? 

The approach taken in **HCA** is very simple from programming point of view. The algorithm starts with the assumption that every individual measurement is a unique cluster. Then it calculates the pairwise distances between the measurements. The two measurements with the smallest distance are grouped together to form the first agglomerative cluster. In the next iteration, the newly generated cluster is then represented by either its mean, minimum or its maximum for the distance calculations. It should be noted that there are several ways to calculate the distance between two measurements (e.g. [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)). For simplicity, we are only going to look at the "Euclidean distance" here.   

In a one dimensional space, the distance between points ``x_{n}`` and ``x_{m}`` is calculated by subtracting the two points from each other.

```math 

d_{m,n} = x_{n} - x_{m}

```

Assuming the below dataset with vectors *X* and *Y* as the coordinates. 

```@example hcax
using AdvChemStat

# Generating the data

cx1 = 1 .+ (2-1) .* rand(5) # 5 random values between 1 and 2 
c1 = 5 .* rand(5)           # 5 random values around 5
cx2 = 4 .+ (6-4) .* rand(5) # 5 random values between 4 and 6
c2 = 10 .* rand(5)          # 5 random values around 10

Y = vcat(c1[:],c2[:])       # building matrix Y
X = vcat(cx1[:],cx2[:])     # building the matrix X

# Plotting the data
scatter(cx1,c1,label = "Cluster 1")
scatter!(cx2,c2,label = "Cluster 2")
xlabel!("X")
ylabel!("Y")

```
We first plot the data in the one dimensional data. In other words, we are setting the y values to zero in our data matrix. Below you can see how this is done.

```@example hcax
using AdvChemStat

# Plotting the data

scatter(X,zeros(length(X[:])),label = "Data")

xlabel!("X")
ylabel!("Y")

```

The next step is to calculate the distances in the x domain. For that we are using the Euclidean distances. Here we need to calculate the distance between each point in the *X* and all the other values in the same matrix. This implies that we will end up with a square distance matrix (i.e. *dist*). The *dist* matrix has a zero diagonal, given that the values on the diagonal represent the distance between each point and itself. Also it is important to note that we are interested only in the magnitude of the distance but not its direction. Thus, you can use the [*abs.(-)*](https://docs.julialang.org/en/v1/base/math/#Base.abs) to convert all the distances to their absolute values. Below you can see an example of a function for these calculations.

```@example hcax
using AdvChemStat

function dist_calc(data)

    dist = zeros(size(data,1),size(data,1))      # A square matrix is initiated 
	for i = 1:size(data,1)-1                     # The nested loops create two unaligned vectors by one member
		for j = i+1:size(data,1)
			dist[j,i] = data[j,1] - data[i,1]    # The generated vectors are subtracted from each other 
		end
	end

	dist += dist'                                # The upper diagonal is filled 
	return abs.(dist)                            # Make sure the order of subtraction does not affect the distances

end 

dist = dist_calc(X)

```

In the next step we need to find the points in the *X* that have the smallest distance and should be grouped together as the first cluster. To do so we need to use the *dist* matrix. However, as you see in the *dist* matrix the smallest values are zero and are found in the diagonal. A way to deal with this is to set the diagonal to for example to *Inf*. 

```@example hcax
using AdvChemStat

#dist = dist_calc(X) 
dist[diagind(dist)] .= Inf                   # Set the diagonal to inf, which is very helpful when searching for minimum distance
dist

```

Now that we have the complete distances matrix, we can use the function [*argmin(-)*](https://docs.julialang.org/en/v1/base/collections/#Base.argmin) to find the coordinates of the points with the minimum distance in the *X*. 

```@example hcax
using AdvChemStat

cl_temp = argmin(dist)

```

The selected points in the *X* are the closest to each other, indicating that they should be grouped into one cluster. In the next step, we will assume this cluster as a single point and thus we can repeat the distance calculations. For simplicity, we assume that the average of the two points is representative of that cluster. This process is called [linkage](https://en.wikipedia.org/wiki/Hierarchical_clustering#Linkage_criteria) and can be done using different approaches. Using the mean, in particular, is called centroid linkage. With centroid method, we are replacing these points with their average. This process can be repeated until all data points are grouped into one single cluster.

```@example hcax
using AdvChemStat

X1 = deepcopy(X)
X1[cl_temp[1]] = mean([X[cl_temp[1]],X[cl_temp[2]]])
X1[cl_temp[2]] = mean([X[cl_temp[1]],X[cl_temp[2]]])

[X,X1]


```

```@example hcax
using AdvChemStat


scatter(X,zeros(length(X[:])),label = "Original data")
scatter!(X1,0.01 .* ones(length(X1[:])),label = "Data after clustering",fillalpha = 0.5)
ylims!(-0.01,0.1)
xlabel!("X")
ylabel!("Y")


```


So far, we have done all our calculations based on one dimensional data. Now we can move towards two and more dimension. One of the main things to consider when increasing the number of dimensions is the distance calculations. In the above examples, we have use the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), which is one of many [distance metrics](https://en.wikipedia.org/wiki/Hierarchical_clustering#Similarity_metric). In general terms the Euclidean distance can be expressed as below, where ``d_{m,n}`` represents the distance between points *m* and *n*. This is based on the [Pythagorean distance](https://en.wikipedia.org/wiki/Euclidean_distance). 

```math
d_{m,n} = \sqrt{\sum{(x_{m} - x_{n})^{2}}}.

```

Let's try to move to a two dimensional space rather than uni-dimensional one. 

```@example hcax
using AdvChemStat

# Plotting the data
scatter(cx1,c1,label = "Cluster 1")
scatter!(cx2,c2,label = "Cluster 2")
xlabel!("X")
ylabel!("Y")

```
The very first step to do so is to convert our 1D distances to 2D ones, using the below equation. If we replace the ``dist[j,i] = data[j,1] - data[i,1]`` with the below equation in our distance function, we will be able to generate the distance matrix for our two dimensional dataset. 

```math
d_{m,n} = \sqrt{(x_{m} - x_{n})^{2} + (y_{m} - y_{n})^{2}}.

```

```@example hcax
using AdvChemStat

function dist_calc(data)

    dist = zeros(size(data,1),size(data,1))      # A square matrix is initiated 
	for i = 1:size(data,1)-1                     # The nested loops create two unaligned vectors by one member
		for j = i+1:size(data,1)
			dist[j,i] = sqrt(sum((data[i,:] .- data[j,:]).^2))    # The generated vectors are subtracted from each other 
		end
	end

	dist += dist'                                # The upper diagonal is filled 
	return abs.(dist)                            # Make sure the order of subtraction does not affect the distances

end 

data = hcat(X,Y)								# To generate the DATA matrix

dist = dist_calc(data) 							# Calculating the distance matrix
dist[diagind(dist)] .= Inf                   # Set the diagonal to inf, which is very helpful when searching for minimum distance
dist

```

```@example hcax
using AdvChemStat

cl_temp = argmin(dist)

data1 = deepcopy(data)

data1[cl_temp[1],1] = mean([data[cl_temp[1],1],data[cl_temp[2],1]])
data1[cl_temp[1],2] = mean([data[cl_temp[1],2],data[cl_temp[2],2]])
data1[cl_temp[2],1] = mean([data[cl_temp[1],1],data[cl_temp[2],1]])
data1[cl_temp[2],2] = mean([data[cl_temp[1],2],data[cl_temp[2],2]])

data1

```

```@example hcax
using AdvChemStat

scatter(data[:,1],data[:,2],label = "Original data")
scatter!(data1[:,1],data1[:,2],label = "Data after clustering")
xlabel!("X")
ylabel!("Y")


```
As it can be seen in the figure, there are two blue points and one red point in the middle of those points. These blue dots represent the two closest data points that are clustered together to form the centroid in between them. If we repeat this process multiple times, we eventually end up having all data points into one large cluster. The HCA clustering generates an array of clustered data points that can be visualized via a [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) or a [heatmap](https://en.wikipedia.org/wiki/Heat_map). 

```@example hcax
using AdvChemStat

dist = dist_calc(data) 

hc = hclust(dist, linkage=:average)
sp.plot(hc)

```

## Practical Application

We can use either our home developed function for HCA or use the julia built-in functions. Here we will provide an easy tutorial on how to use the julia functions that are built-in the **AdvChemStat.jl** package. 

For calculating the distances the function *pairwise(-)* via the julia package [**Distances.jl**](https://github.com/JuliaStats/Distances.jl) can be used. Function *pairwise(-)* has three inputs namely: 1) distance metrics, 2) data, and 3) the operation direction. This function outputs a square matrix similar to our distance matrix. As it can be seen from the distance matrix, our function and the *pairwise(-)* generate the same results, which is expected. The function *pairwise(-)* will give access to a wide variety of distance metrics that can be used for your projects. 

```@example hcax
using AdvChemStat

dist1 = pairwise(AdvChemStat.Euclidean(), data, dims=1) # Euclidean distance 

# dist1 = pairwise(AdvChemStat.TotalVariation(), data, dims=1) # TotalVariation distance 

```

For the HCA itself, you can use the function *hclust(-)* incorporated in the **AdvChemStat.jl** package and provided via [**Clustering.jl**](https://juliastats.org/Clustering.jl/stable/index.html) package. This function takes two inputs, the distance matrix and the linkage method. The output of this function is a structure with four [outputs](https://juliastats.org/Clustering.jl/stable/hclust.html#Clustering.Hclust). The two most important outputs are *merges* and *order*. The combination of all four outputs can be plotted via *sp.plot(-)* function. 

```@example hcax
using AdvChemStat

h = hclust(dist1,:average) # Average linkage or centroids 

```
To access the outputs, one can do the following: 
```@example hcax
using AdvChemStat

h.order 

```
and to plot the outputs, we can use the below function.

```@example hcax
using AdvChemStat

sp.plot(h)

```

There are also *python* implementation of [HCA](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), that you can explore using those for your analysis. 

## Additional Example

If you are interested in practicing more, you can use the [mtcars](https://github.com/JuliaStats/RDatasets.jl/blob/master/doc/datasets/rst/mtcars.rst) dataset via **RDatasets** provided in [folder dataset](https://github.com/EMCMS/AdvChemStat.jl/tree/main/datasets) of the package [*AdvChemStat.jl* github repository](https://github.com/EMCMS/AdvChemStat.jl). Please note that you must exclude the car origin column. The objective here is to see whether HCA is able to cluster the cars with similar origins.  

If you are interested in additional resources regarding **HCA** and would like to know more you can check this [MIT course material](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/resources/lecture-12-clustering/).  