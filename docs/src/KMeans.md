# K-means Clustering

## Introduction

The [**k-means clustering**](https://en.wikipedia.org/wiki/K-means_clustering) is an unsupervised clustering algorithm that uses the distance metrics to create the user defined number of clusters. The **k-means** can be used for both continuous and discreet data, given that there are at least two variables provided. The number of clusters to be formed are typically provided by the user and is typically based on the *prior* knowledge. The center of each cluster in **k-means** is called *centroid* and meant to represent that cluster of data. The **k-means** algorithm is considered a greedy algorithm as at the end every single point in the dataset is part of a specific cluster. Also, **k-means** is an iterative algorithm , given that the process of finding the local minimum is done over multiple iterations. The algorithmic details of **k-means** are provided below. 

## How? 

In **k-means** the algorithm initialized with a set of randomly selected centroids. The number of these centroids is user defined while their location is randomly selected. Based on the [Euclidean distances](https://en.wikipedia.org/wiki/Euclidean_distance) of each point in the dataset and the centroids, the points are assigned to each cluster. Here *l* defines the number of variables, *m* is the number of points, *c_{n}* is the number of the clusters, and *x* is the coordinates on each point/centroid. It should be noted that *m*, *n*, and *c* all must be real numbers and larger than 0. Additionally, the condition of *m* > *n* must be fulfilled. This means that the number of points must be larger than the number of clusters.    

```math 

d_{d,c_{n}} = \sqrt{\sum _{i=1}^{l} (x_{m} - x_{c_n})^2}

```
After assigning the data points to the first set of temporary clusters, the **k-means** algorithm adjusts the centroids by putting them in the center of clusters, only considering the actual data. At this stage the process of distance calculation is repeated again and the points may be reassigned to another cluster based on their distances. This process is repeated until either the location of centroids remain constant or there is no reassignment of the points. These are typical stopping signals for the **k-means** algorithm. 
 

## Practical Example 

Let's build a simple **k-means** algorithm together using some random data: 


```@example km
using AdvChemStat

c1 = 5 .* rand(5)
cx1 = 1 .+ (2-1) .* rand(5)
c2 = 20 .* rand(5)
cx2 = 6 .+ (10-6) .* rand(5)

data_s = vcat(hcat(cx1,c1),hcat(cx2,c2))

```
```@example km

scatter(data_s[:,1],data_s[:,2],label=false)
xlabel!("X")
ylabel!("Y")

```

### Step 1: Selection of centroids

For this case we have only two variables in our dataset *data_s*. For simplicity, we will start with two clusters, thus *k* is equal 2. In this case, we can either select two random points within the measurements' window (i.e. between minimum and maximum of values in *data_s*) or we can choose the indices of two of the measurements at random. Here we go with the second option.  

```@example km
k = 2 # k is the number of clusters

ind_c = Int.(round.(1 .+ (size(data_s,1) - 1) .* rand(k))) # index of the centroids

```

```@example km
scatter(data_s[:,1],data_s[:,2],label="data_s",legend=:topleft)
scatter!(data_s[ind_c,1],data_s[ind_c,2],label="Centroids")
xlabel!("X")
ylabel!("Y")


```

### Step 2: Calculation of distances

To calculate the distances, we create a vector containing the distance of each centroid from every single point in the *data_s*. This will result in a distance matrix of ``d_{10 \times 2}``, where column one belongs to the first centroid and the first point corresponds to the first data point in *data_s*. Since we are using the actual measurements as our starting point, we will have zero distances for at least one point per column.

```@example km

d = zeros(size(data_s,1),length(ind_c)) # Generating the distance matrix

for i = 1:length(ind_c)
    cent = transpose(data_s[ind_c[i],:])
    d[:,i] = sqrt.(sum((cent .- data_s).^2,dims=2))

end 

d

```

### Step 3 Assigning the points to each cluster

Now that we have our distance matrix *d*, we need to assign each point to a cluster, in our case the two clusters. To do so, we can look at our *d* matrix row wise and based on the column with the minimum distance assign that point to a cluster. For example, if in *d[1,:]* the minimum value is located at *d[1,2]*, then this point belongs to the cluster number two. These calculations can be done using the following code. 

```@example km

clusters = zeros(size(d))

for i = 1:size(d,1)
    clusters[i,argmin(d[i,:])] = argmin(d[i,:])
end 
        

clusters

```

```@example km
scatter(data_s[clusters[:,1] .>0,1],data_s[clusters[:,1] .>0,2],label="Cluster 1",legend=:topleft)
scatter!(data_s[clusters[:,2] .>0,1],data_s[clusters[:,2] .>0,2],label="Cluster 2")
scatter!(data_s[ind_c,1],data_s[ind_c,2],label="Centroids")
xlabel!("X")
ylabel!("Y")


```

### Step 4 Adjusting the centroids 

Now that we have the first set of temporary clusters, we need to recalculate the centroids using the the data points that are assigned to a specific cluster. To do that we calculate the mean of *data_s* column wise where the values in the *clusters* matrix is different from zero. 


```@example km
cents = zeros(size(clusters,2),size(data_s,2))

for i=1:size(clusters,2)

    tv = clusters[:,i]
    #println(tv)
    cents[i,:] = mean(data_s[tv .>0,:],dims=1)

end 

cents

```

```@example km
scatter(data_s[clusters[:,1] .>0,1],data_s[clusters[:,1] .>0,2],label="Cluster 1",legend=:topleft)
scatter!(data_s[clusters[:,2] .>0,1],data_s[clusters[:,2] .>0,2],label="Cluster 2")
scatter!(data_s[ind_c,1],data_s[ind_c,2],label="Centroids")
scatter!(cents[:,1],cents[:,2],label="New centroids")
xlabel!("X")
ylabel!("Y")


```
As you can see, the locations of the "New centroids" have changed. Depending on the starting points, the new locations may or may not be closer to the global minimum of the system. The reach such a point, we need to repeat this process multiple times until we do not see any changes in the location of the centroids and thus the reassignment of the points to different clusters. 


## Applications 

The **k-means** algorithm is mainly used for the clustering of multivariate data. Typically, it follows an HCA or PCA to identify the number of clusters. Then that number is fed to the **k-means** to generate the clusters. When dealing with very large number of variables, **k-means** is not the best algorithm as it may converge to a local minimum rather than a global minimum. It is recommended to run the **k-means** algorithms multiple times to make sure about the robustness of the location of the centroids. In fact most of existing packages for **k-means** have this feature already built in them. 

### Packages 

There are different implementations of **k-means** algorithm available through **AdvChemStat.jl** package. Below the two main implementations are discussed. 

#### [Clustering.jl](https://juliastats.org/Clustering.jl/stable/kmeans.html#) 

The function *kmeans(-)* is provided via **AdvChemStat.jl**. Please note that the matrix fed to the *kmeans(-)* should have the variables in rows and measurements in columns. This means that for making it work with our usual datasets, you need to transpose your matrix prior to the model building.   

```@example km

k_mod = kmeans(transpose(data_s),2) # to build a k-means model

```
Once, the model is built, you are able to get the points assigned to each cluster as well as the coordinates of the centroids. It should be noted that the coordinates of the centroids, similarly to our data are transposed. Thus each represents the coordinates for each centroid rather than the columns. 

```@example km

a_m = assignments(k_mod) # to get the assignment of the points in the data based on the model
c_c = k_mod.centers


```

We can also plot our results for visualization.

```@example km
scatter(data_s[:,1],data_s[:,2],group=k_mod.assignments,legend=:topleft) 
scatter!(c_c[1,:],c_c[2,:],label="Centroids")
xlabel!("X")
ylabel!("Y")

```

It should be noted that the julia implementation of **k-means** algorithm does not have an *apply(-)* incorporated in it meaning that for new data the distances from centroids must be calculated manually for the cluster assignment. This is in-line with the unsupervised nature of the **k-means** algorithm that should not be used for inferences. 

#### [sklearn.cluster.k_means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) 

Through **AdvChemStat.jl** package you can also use the python implementation of **k-means** via **scikit-learn**. In this case there is no need for transposing your data. Thus you can use your data as it is (i.e. columns for variables and rows for the measurements). The algorithm outputs a python object containing the coordinates of the centroids and the assigned cluster to each point. Here is an example for our *data_s*.

```julia 
using AdvChemStat

@sk_import cluster: KMeans

kmeans_ = KMeans(n_clusters=2, random_state=0).fit(data_s)


kmeans_.cluster_centers_
```
We can also plot the results as shown below. Please note that in case of python implementation you can use the function *apply(-)* to perform prediction using the built model.


## Additional Example

If you are interested in practicing more, you can use the [NIR.csv](https://github.com/EMCMS/AdvChemStat.jl/blob/main/datasets/NIR.csv) file provided in the [folder dataset](https://github.com/EMCMS/AdvChemStat.jl/tree/main/datasets) of the package [*AdvChemStat.jl* github repository](https://github.com/EMCMS/AdvChemStat.jl). You can try to use the NIR spectra and **k-means** to see whether there are clear clusters of samples associated with different octane number. 

If you are interested in additional information about **k-means** and would like to know more you can check this [MIT course material](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/resources/lecture-12-clustering/).  