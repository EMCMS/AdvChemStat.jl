# Singular Value Decomposition (SVD) 

## Introduction

The [**SVD**](https://en.wikipedia.org/wiki/Singular_value_decomposition) is a matrix factorization technique that decomposes any matrix to a unique set of matrices. The **SVD** is used for dimension reduction, trend analysis, and potentially for the clustering of a multivariate dataset. **SVD** is an exploratory approach to the data analysis and therefore it is an unsupervised approach. In other words, you will only need the *X* block matrix. However, where the *Y* matrix/vector is available, it (i.e. *Y*) can be used for building composite models or assess the quality of the clustering. 

## How? 

In **SVD** the matrix *``X_{m \times n}``* is decomposed into the matrices *``U_{m \times n}``*, *``D_{n \times n}``*, and *``V_{n \times n}^{T}``*. The matrix *``U_{m \times n}``* is the left singular matrix and it represents a rotation in the matrix space. The *``D_{n \times n}``* is diagonal matrix and contains the singular values. This matrix may be indicated with different symbols such as *``\Sigma_{n \times n}``*. The *``D_{n \times n}``* matrix in the geometrical space represents an act of stretching. Each *singular value* is the degree and/or weight of stretching. We use the notation *``D_{n \times n}``* to remind ourselves that this is a diagonal matrix. Finally, *``V_{n \times n}^{T}``* is called the right singular matrix and is associated with rotation. Overall, **SVD** geometrically is a combination of a rotation, a stretching, and a second rotation.

The two matrices *``U_{m \times n}``* and *``V_{n \times n}^{T}``* are very special due to their [unitary](https://en.wikipedia.org/wiki/Unitary_matrix) properties.

```math 

U^{T} \times U = U \times U^{T} = I\\
V^{T} \times V = V \times V^{T} = I

```

Therefore the general matrix expression of **SVD** is the following: 

```math
X = UDV^{T}.

```
To deal with the non-square matrices, we have to convert our *X* matrix to ``X^{T} \times X``. This implies that our **SVD** equation will become the following: 

```math
X^{T}X = (UDV^{T})^{T} \times UDV^{T}.

```
And after a little bit of linear algebra: 

```math
X^{T}X = VD^{T} \times DV^{T} \\ 
and \\

XV = UD.

```
This is a system of two equations with two variables that can be solved. Before looking at an example of such system let's remind ourselves that ``VD^{T} \times DV^{T}`` is the solution of [eigenvalue/eigenvector decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) of ``X^{T}X``. This means that both *D* and *``V^{T}``* can be calculated by calculating the [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of ``X^{T}X``. Therefore we can calculate *D* and *V* as follows:

```math

D = \sqrt{eigenvalues(X^{T}X)}\\ 
V = eigenvector(X^{T}X).

```

Once we know *V*, we can use that and the second equation of SVD to calculate the last part i.e. the matrix *U*. 

```math
U = XVD^{-1} 

```
Please note that ``D^{-1}`` denotes the [inverse or pseudo-inverse](https://en.wikipedia.org/wiki/Invertible_matrix) of the matrix *D*.  

## Practical Example 

Let's do the **SVD** calculations together for the below matrix: 


```@example svdex
using AdvChemStat

X = [5 -5;-1 7;1 10]

```

### Step 1: ``X^{T}X``

```@example svdex
# The function transpose(-) is part of LinearAlgebra.jl package that has been automatically installed via AdvChemStat.jl package.
# Not all the functions of LinearAlgebra.jl are exported within the AdvChemStat.jl environment. 
XtX = transpose(X)*X 

```

### Step 2: Calculation of *D*, *V*, and *U*

```@example svdex

D = diagm(sqrt.(eigvals(XtX))) # A diagonal matrix is generated

```

```@example svdex

V = eigvecs(XtX) # Right singular matrix


```

```@example svdex

U = X*V*pinv(D)	# Left singular matrix


```

#### Builtin Function

The same calculations can be done with the function *svd(-)* of AdvChemStat package provided via LinearAlgebra.jl package. 

```@example svdex

 out = svd(X)

```

```@example svdex

 D = diagm(out.S) # The singular value matrix

```

```@example svdex

 U = out.U # Left singular matrix

```

```@example svdex

 V = transpose(out.Vt) # Right singular matrix

```

Please note that the builtin function sorts the singular values in descending order and consequently the other two matrices are also sorted following the same. Additionally, for ease of calculations the builtin function generates the mirror image of the *U* and *V* matrices. These differences essentially do not impact your calculations at all, as long as they are limited to what is listed above.

### Step 3 Calculation of ``\hat{X}``

Using both the manual method and the builtin function, you can calculate ``\hat{X}`` following the below operation. 

```@example svdex

X_hat = U*D*transpose(V)

```
## Applications 

As mentioned above **SVD** has several applications in different fields. Here we will focus on three, namely: dimension reduction, clustering/trend analysis, and multivariate regression. This dataset contains five variables (i.e. columns) and 150 measurements (i.e. rows). The last variable "Species" is a categorical variable which defines the flower species. 

### Dimension Reduction

To show case the power of **SVD** in dimension reduction we will use the *Iris* dataset from [Rdatasets](https://github.com/JuliaStats/RDatasets.jl). 

```@example iris
using AdvChemStat

data = dataset("datasets", "iris")
describe(data) # Summarizes the dataset

```

Here we show how **SVD** is used for dimension reduction with the *iris* dataset. First we need to convert the dataset from table (i.e. [dataframe](https://dataframes.juliadata.org/stable/)) to a matrix. For data we can use the function *Matrix(-)* builtin in the julia core language.

```@example iris
Y = data[!,"Species"]
X = Matrix(data[:,1:4]); # The first four columns are selected for this

```

Now we can perform **SVD** on the *X* and try to assess the underlying trends in the data. 

```@example iris

 out = svd(X)

```

```@example iris

 D = diagm(out.S) # The singular value matrix

```

```@example iris

 U = out.U # Left singular matrix

```

```@example iris

 V = transpose(out.Vt) # Right singular matrix

```

As you may have noticed, there are four variables in the original data and four non-zero singular values. Each column in the lift singular matrix is associated with one singular value and one row in the *V* matrix. For example the first column of sorted *U* matrix (i.e. via the builtin function) is directly connected to the first singular value of 95.9 and the first row of the matrix *V*. With all four singular values we can describe 100% of variance in the data (i.e. ``\hat{X} = X``). By removing the smaller or less important singular values we can reduce the number of dimensions in the data. We can calculate the variance explained by each singular value via two different approaches. 

```@example iris

 var_exp = diag(D) ./ sum(D) # diag() selects the diagonal values in a matrix 


```

```@example iris

 var_exp_cum = cumsum(diag(D)) ./ sum(D) # cumsum() calculates the cumulative sum 


```

```@example iris

 scatter(1:length(var_exp),var_exp,label="Individual")
 plot!(1:length(var_exp),var_exp,label=false)

 scatter!(1:length(var_exp),var_exp_cum,label="Cumulative")
 plot!(1:length(var_exp),var_exp_cum,label=false)
 xlabel!("Nr Singular Values")
 ylabel!("Variance Explained")


```
Given that the first two singular values explain more than 95% variance in the data, they are considered enough for modeling our dataset. The next step here is to first plot the scores (i.e. the left singular matrix) of first and second singular values against each other to see whether we have a model or not. Each column in the *``U``* matrix represents a set of scores associated with a singular value (e.g. first column for the first singular value).

```@example iris

 scatter(U[:,1],U[:,2],label=false)
 xlabel!("First Singular value (81%)")
 ylabel!("Second Singular value (15%)")
 

```
At this point we are assuming that we do not have any idea about the plant species included in our dataset. Now we need to connect the singular values to individual variables. For that similarly to [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) we will take advantage of the loadings, which in this case are the columns of the *``V``* or the rows of *``V^{T}``*. 

```@example iris

 bar(V[:,1] ./ sum(abs.(V)),label="First SV")
 bar!(V[:,2] ./ sum(abs.(V)),label="Second SV")
 xlabel!("Variable Nr")
 ylabel!("Importance")
 #ylims!(-0.1,0.1)

 

```
The sign of each loading value shows the relationship between the variable and the model. For example, based on the first *SV* the variable number one and two both have a negative impact on the final model (i.e. scores of the *SV1*). A positive impact indicates an increase of the final model scores with the variable while a negative impact means a decrease in the score values with an increase the variable. 

```@example iris

 p1 = scatter(X[:,1],U[:,1],label=false)
 xlabel!("SepalLength")
 ylabel!("Scores U1")

  p2 = scatter(X[:,2],U[:,1],label=false)
 xlabel!("SepalWidth")
 ylabel!("Scores U1")

  p3 = scatter(X[:,2],U[:,2],label=false)
 xlabel!("SepalWidth")
 ylabel!("Scores U2")

 p4 = scatter(X[:,3],U[:,2],label=false)
 xlabel!("PetalLength")
 ylabel!("Scores U2")

 plot(p1,p2,p3,p4,layout = (2,2))
 

```
In this particular case, the *SV1* is a linear combination of *SepalLength* and *SepalWidth* while the *SV2* is a linear combination of all four variables. This implies that we can cover the variance present in the ``X`` with two variables, which are ``U1`` and ``U2``. For this dataset, we have a reduction of variables from 4 to 2, which may not look impressive. However, this can be a very useful technique when dealing with a large number of variables (the octane example).

### Clustering 

When we perform cluster analysis or most modeling approaches, we need to divide our data into training and test sets. We usually go for a division of 80% for training set and 20% for the test. More details are provided in the cross-validation chapter. Let's randomly select 15 data points to put aside as the test set. 

```@example iris

n = 15 # number of points to be selected

rand_ind = rand(1:size(X,1),n) # generate a set of random numbers between 1 and size(X,1)
ind_tr = ones(size(X,1))       # generate a matrix of indices 
ind_tr[rand_ind] .= 0          # set the test set values' indices to zero 
X_tr = X[ind_tr .== 1,:]       # select the training set
X_ts = X[rand_ind,:]           # select the test set
data[rand_ind,:]


```
Now that we have training and test sets separated, we can build our model using the training set. This implies that the model has never seen the values in the test set. It should be noted that we always want the homogenous distribution of measurements in the test set. Also, each iteration here will result in a different test set as a new set of random numbers are generated. 
Now let's build our model with only the ``X_{tr}`` following the same procedure as before. 

```@example iris

 out = svd(X_tr)

```

```@example iris

 D = diagm(out.S) # The singular value matrix

```

```@example iris

 U = out.U # Left singular matrix

```

```@example iris

 V = transpose(out.Vt) # Right singular matrix

```
Let's plot our results for the first two **SVs**, as we did before. However, this time we will take the knowledge of the different species into account. 

```@example iris

 var_exp = diag(D) ./ sum(D) # variance explained 
 Y_tr = data[ind_tr .== 1,"Species"]
 Y_ts = data[ind_tr .== 0,"Species"]
 scatter(U[:,1],U[:,2],label=["Setosa" "Versicolor" "Virginica"], group = Y_tr)
 xlabel!("First Singular value (81%)")
 ylabel!("Second Singular value (14%)")


```
As it can be seen, this model is very similar to our previous model based on the full dataset. Now we need to first define thresholds for each class based on the score values in the ``U1`` and ``U2`` space. This is typically more difficult to assess. However, for this case the main separating factor is the ``U2`` values (e.g. ``U2 \geq 0.05 = Setosa``). 


```@example iris

 scatter(U[:,1],U[:,2],label=["Setosa" "Versicolor" "Virginica"], group = Y_tr)
 plot!([-0.15,0],[0.05,0.05],label="Setosa")
 plot!([-0.15,0],[-0.04,-0.04],label="Virginica")
 xlabel!("First Singular value (81%)")
 ylabel!("Second Singular value (14%)")


```

The next step is to calculate the score values for the measurements in the test set. This will enable us to estimate the class associated with each data point in the test set. To do this we need to do a little bit of linear algebra.

```math
X = UDV^{T}\\

U_{test} = X \times (DV^{T})^{-1}

```
In practice:

```@example iris

 U_test = X_ts * pinv(D*transpose(V))


```
```@example iris

 scatter(U[:,1],U[:,2],label=["Setosa" "Versicolor" "Virginica"], group = Y_tr)
 plot!([-0.15,0],[0.05,0.05],label="Setosa")
 plot!([-0.15,0],[-0.04,-0.04],label="Virginica")
 scatter!(U_test[Y_ts[:] .== "setosa" ,1],U_test[Y_ts[:] .== "setosa",2],label="Setosa",marker=:d)
 scatter!(U_test[Y_ts[:] .== "versicolor",1],U_test[Y_ts[:] .== "versicolor",2],label="Versicolor",marker=:d)
 scatter!(U_test[Y_ts[:] .== "virginica",1],U_test[Y_ts[:] .== "virginica",2],label="Virginica",marker=:d)
 xlabel!("First Singular value (81%)")
 ylabel!("Second Singular value (14%)")


```

As it can be seen from the results of the test set, our model is not prefect but it does well for most cases. It should be noted that steps such as data pre-treatment and the use of supervised methods may improve the results of your cluster analysis. The use of **SVD** for prediction is not recommended. It must be mainly used for the dimension reduction and data exploration.  

### Regression

If you have a dataset (e.g. octane dataset in the additional example), where the **SVD** is used to reduce the dimensions of the dataset. In this case the we can perform a least square regression using the selected columns of ``U`` rather than the original ``X``. For example in case of *iris* dataset the ``U1`` and ``U2`` can be used to replace ``X``.  

```julia 
 X_svr = U[:,1:m] # m is the number of selected SVs 
 Y_svr            # does not exist for iris dataset. for the octane dataset is the octane column
 b = pinv(transpose(X_svr) * X_svr) * transpose(X_svr) * Y_svr # simple least square solution
 y_hat = X_svr * b # prediction the y_hat 
```

### Trend Analysis

We also can assess the trend represented by each *SV* in our model. This is typically done by setting all *SV* values except one to zero. Then the new ``D`` is used to predict ``\hat{X}``. Then different variables are plotted against each other for both ``X`` matrices. 

```@example iris
 D_temp = out.S
 D_temp[2:end] .= 0
 D_n = diagm(D_temp) # the new singular value matrix

```
Then the ``\hat{X}`` is calculated.

```@example iris
 X_h  = U * D_n * transpose(V)
 X_h[1:5,:]

```
Now if we plot the SepalLength vs SepalWidth we can clearly see a clear 1 to 2 relationship between the two variables which is being detected by the first *SV*. This can be done for other variables and *SVs*. 
```@example iris

 scatter(X[:,1],X[:,2],label="X")
 scatter!(X_h[:,1],X_h[:,2],label="X_h")
 xlabel!("SepalLength")
 ylabel!("SepalWidth")


```

## Additional Example

If you are interested in practicing more, you can use the [NIR.csv](https://github.com/EMCMS/AdvChemStat.jl/blob/main/datasets/NIR.csv) file provided in the [folder dataset](https://github.com/EMCMS/AdvChemStat.jl/tree/main/datasets) of the package [*AdvChemStat.jl* github repository](https://github.com/EMCMS/AdvChemStat.jl). Please note that this is an *SVR* problem, where you can first use **SVD** for the dimension reduction and then use the selected *SVs* for the regression. 

If you are interested in math behind **SVD** and would like to know more you can check this [MIT course material](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-6-singular-value-decomposition-svd/).  