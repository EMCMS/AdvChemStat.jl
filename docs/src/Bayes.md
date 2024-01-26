# Bayesian Statistics  

## Introduction

[**Bayesian Statistics**](https://en.wikipedia.org/wiki/Bayesian_statistics) is a statistical approach based on the **Bayes Theorem**, where the probability of an event is expressed as the degree of belief. A major difference between **Bayesian Statistics** and the [**frequentist**](https://en.wikipedia.org/wiki/Frequentist_probability) approach is the inclusion of the prior knowledge of the system into the probability calculations. One of the main differences between **Bayesian Statistics** and conventional [**t-test**](https://en.wikipedia.org/wiki/Student%27s_t-test) or [**ANOVA**](https://en.wikipedia.org/wiki/Analysis_of_variance) is that the **Bayesian Statistics** considers the probability of both sides of the [null-hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis). 

## Bayes Theorem 

In **Bayes Theorem** the probability of an even occurring is updated by the degree of belief (i.e. prior knowledge). The **Bayes Theorem** is mathematically express as follows: 

```math 

P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}.

```

The term *``P(A \mid B)``* is the posterior probability of *A* given *B*, implying that *A* and *B* are true given that *B* is true. The second term in this formula is the conditional probability of *``P(B \mid A)``*. The term *P(A)* is defined as the *prior probability*, which enables the incorporation of prior knowledge into the probability distribution of even occurring. Finally, *P(B)* is the marginal probability, which is used as a normalizing factor in this equation and it must be > 0. To put these terms into context, let's look at the following example. 

There is a new test for the detection of a new variant of COVID-19. We want to calculate the probability of a person being actually sick given a positive test (i.e. the posterior probability *``P(A \mid B)``*). The conditional probability (*``P(B \mid A)``*) in this case is the rate of true positive for the test. In other words the percentage of sick people who tested positive. The prior probability (*P(A)*) in this case is the probability of people getting sick while the marginal probability (*P(B)*) is all people who test positive independently from their health status.   

### Bayes Formula

The derivation of **Bayes Theorem** is very simple and can be done with simple knowledge of probability and arithmetics. Let's first write the two conditional probabilities (i.e. the posterior and conditional probabilities) as functions of their [joint probabilities](https://en.wikipedia.org/wiki/Joint_probability_distribution). 

```math 

P(A \mid B) = \frac{P(A \cap B)}{P(B)} \\
P(B \mid A) = \frac{P(B \cap A)}{P(A)}

```
Given that ``P(A \cap B) = P(B \cap A)`` and is is unknown, one can solve the above equations as a function of this unknown joint probability. 

```math 

P(A \cap B) = P(A \mid B)P(B) \\
P(A \cap B) = P(B \mid A)P(A)

```

This means that the right sides of these equations can be assumed equal as the left sides are equal, thus: 

```math 

P(A \mid B)P(B) = P(B \mid A)P(A).

```
Now if we divide both sides of this equation by *P(B)*, we will end up with the **Bayes Theorem**.

```math 

P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}.

```

## Practical Example

This is a very simple example where we work together to calculate the joint probabilities which are needed for the conditional probability calculations. 

We have a classroom with 40 students. We asked them to fill out a survey about whether they like football, rugby, or none of them. The results of our survey show that 30 students like football, 10 students like rugby, and 5 do not like these sports. As you can see, the total number of votes is larger than 40, implying that 5 students like both sports.  

```@example bayes
using AdvChemStat

plot([1,1],[1,6],label=false,color = :blue)
plot!([1,9],[1,1],label=false,color =:blue)
plot!([9,9],[1,6],label=false,color =:blue)
plot!([1,9],[6,6],label="All students",color =:blue)
plot!([3,3],[1,6],label=false,color =:red)
plot!([1,3],[3.5,3.5],label=false,color =:green)
plot!([3,5],[3.5,3.5],label=false,color =:orange)
plot!([5,5],[3.5,6],label=false,color =:orange)
annotate!(2,5,"Rugby Only")
annotate!(4,5,"Both")
annotate!(2,2,"None")
annotate!(6,2,"Football Only")

```

Now lets generate the associated [contingency table](https://en.wikipedia.org/wiki/Contingency_table) based on the survey results. Our table is a two by two one, given the number of questions asked. The contingency table will help us to calculate the joint probabilities. The structure of our table will be the following:

|    | Y Football | N Football|
|:--- | :---: | ---: |
|Y Rugby |    |      |
|N Rugby |    |      |

After filling the table with the correct frequencies, we will end up with the following: 

|    | Y Football | N Football|
|:--- | :---: | ---: |
|Y Rugby |  5  |   5   |
|N Rugby |  25  |   5   |

These numbers can also be expressed in terms of probabilities rather than frequencies. 

|    | Y Football | N Football|
|:--- | :---: | ---: |
|Y Rugby |  5/40 = 0.125  |   5/40 = 0.125   |
|N Rugby |  25/40 = 0.625  |   5/40 = 0.125   |

We can now further expand our table to include total cases of football and rugby fans amongst the students. 

|    | Y Football | N Football| Rugby total|
|:--- | :---: | :---: | ---: |
|Y Rugby |  5/40 = 0.125  |   5/40 = 0.125   | 10/40 = 0.25 |
|N Rugby |  25/40 = 0.625  |   5/40 = 0.125   | 30/40 = 0.75 |
Football total | 30/40 = 0.75 | 10/40 = 0.25 | |


We can use the same table for calculating conditional probabilities, for example *``P(A \mid B)``* or expanded to *``P(A \& B \mid B)``*. Let's remind ourselves the formula for this: 

```math 

P(A \mid B) = \frac{P(A \cap B)}{P(B)}.

```
For this example we want to calculate the probability of a student liking football given that they are carrying a rugby ball (i.e. they like rugby). So now we can write the above formula using the annotation related to this question. 

```math 

P(Football \& Rugby \mid Rugby) = \frac{P(Football \cap Rugby)}{P(Total Rugby)}.

```

Now we can plug in the numbers from our contingency table to calculate the needed probability. 

```math 

P(Football \& Rugby \mid Rugby) = \frac{0.125}{0.25} = 0.5

```

Another way to express these probabilities is using the [probability trees](https://www.mathsisfun.com/data/probability-tree-diagrams.html). Each branch in a tree represents one set of conditional probabilities.   


## Applications

**Bayesian Statistics** has several applications from [uncertainty assessment](https://en.wikipedia.org/wiki/Monte_Carlo_method) to [network analysis](https://en.wikipedia.org/wiki/Bayesian_network) as well as simple regression and classification. Here we will discuss  classification (i.e. [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)), uncertainty assessment, and regression. 

### Naive Bayes

Imagine you are measuring the concentration of a certain pharmaceutical in the urine samples of a clinical trial. This drug at low concentrations (i.e. < 5 ppb) does not have an effect while at high levels (i.e. > 9) could be lethal. Your previous measurements of these concentrations results in the below distribution.   

```@example bayes
using AdvChemStat

x = [2.11170571,4.654611177,2.058684377,9.118890998,6.482271164,1.741767743,0.423550831,3.930361297
,8.394899978,2.720184918,4.642679068,0.698396604,10.60195845,9.949609087,9.788688087,9.275078609
,3.71104968,3.048191598,7.131314198,2.696493503]

histogram(x,bins=5,normalize=:probability,label=false)
xlabel!("Concentration (ppb)")
ylabel!("Probability")

```

If we overlay the two concentration distributions assuming [Normality](https://en.wikipedia.org/wiki/Normal_distribution), we will end up with the below figure. 

```@example bayes
using AdvChemStat

histogram(x,bins=5,normalize=:probability,label=false)
plot!(Normal(9.5,0.7*std(x)),label="Toxic")
plot!(Normal(2.5,std(x)),label="No effect",c=:black)

xlabel!("Concentration (ppb)")
ylabel!("Probability")

```

Now you are given a new sample to measure for the concentration of the drug. Your boss is only interested in whether this new sample is a no effect or a toxic case. After your measurement you will end up with the below results. 

```@example bayes
using AdvChemStat

histogram(x,bins=5,normalize=:probability,label=false)
plot!(Normal(9.5,0.7*std(x)),label="Toxic")
plot!(Normal(2.5,std(x)),label="No effect",c=:black)
plot!([8.5,8.5],[0,0.4],label="Measurement",c=:blue)

xlabel!("Concentration (ppb)")
ylabel!("Probability")

```

When looking at your results, you intuitively will decide that this is a case of toxic levels. This is done by comparing the distance from the measurement to the apex of each distribution. 

```@example bayes
using AdvChemStat

histogram(x,bins=5,normalize=:probability,label=false)
plot!(Normal(9.5,0.7*std(x)),label="Toxic",c=:red)
plot!(Normal(2.5,std(x)),label="No effect",c=:black)
plot!([8.5,8.5],[0,0.4],label="Measurement",c=:blue)

plot!([8.5,9.5],[0.1675,0.1675],label="d1",c=:red)

plot!([2.5,8.5],[0.1172,0.1172],label="d2",c=:black)

xlabel!("Concentration (ppb)")
ylabel!("Probability")

```

Performing such assessments visually is only possible when we are dealing with very clear-cut cases with limited number of dimensions and categories. When dealing with more complex systems, we need to have a more clear metrics to assign a sample to a specific group of measurements. To generate such metrics we can calculate the posterior probability of the measurement being part of a group, given a prior distribution (i.e. the distribution of each group). We can calculate these posterior probabilities via *pdf(-)* of **AdvChemStat.jl** package (implemented through package [Distributions.jl](https://juliastats.org/Distributions.jl/stable/)). So in this case we can perform these calculations as follows:

```@example bayes
using AdvChemStat

PT = pdf(Normal(9.5,0.7*std(x)),8.5)
PnE = pdf(Normal(2.5,std(x)),8.5)
P = [PT,PnE]

```


```@example bayes
using AdvChemStat

histogram(x,bins=5,normalize=:probability,label=false)
plot!(Normal(9.5,0.7*std(x)),label="Toxic")
plot!(Normal(2.5,std(x)),label="No effect",c=:black)
plot!([8.5,8.5],[0,0.4],label="Measurment",c=:blue)
annotate!(6.5,0.28,string(round(PnE,digits=2)))
annotate!(10,0.28,string(round(PT,digits=2)))

xlabel!("Concentration (ppb)")
ylabel!("Probability")

```

These calculations clearly show that the posterior probability of toxic level is larger than the one for the no-effect concentrations resulting in the classification of this new measurement. If dealing with more than one variable, then the product of the posterior probability of each variable will give you the multivariate posterior probability of a measurement being part of a class, given the prior distributions. 

```math 

P(A \mid B_{1:n}) = \prod _{i =1}^{n} P(A \mid B_{i})

```

### Uncertainty Assessment

Let's assume that the figure above is related to the several measurements of the same drug in the wastewater. Our objective here is to estimate what the true estimation of the drug concentration in our samples is. The usual procedure here is to first assume a normal distribution calculating the mean and standard deviation of the data. 

```@example bayes
using AdvChemStat

histogram(x,bins=5,normalize=:probability,label=false)
plot!(Normal(mean(x),std(x)),label="Distribution assuming normality")

xlabel!("Concentration (ppb)")
ylabel!("Probability")

```
This is clearly not the best distribution describing our dataset. Another strategy is a [nonparametric](https://en.wikipedia.org/wiki/Nonparametric_statistics) one i.e. [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)). In this case we try to assess the distribution of mean and the standard deviation of the measurements. For example we can sample 85% of the *X* over each iteration, enabling the calculation of the mean and the standard deviation of the measurements without the normality assumption. 

```@example bayes
using AdvChemStat

function r_sample(x,n,prc)
    m = zeros(n)
    st = zeros(n)

    for i =1:n 
        tv1 = rand(x,Int(round(prc*length(x))))
        m[i] = mean(tv1)
        st[i] = std(tv1)

    end 

    return m, st 
    
end


n = 10000
prc = 0.85

m, st = r_sample(x,n,prc)

```

Now if we plot these distributions we can see that the mean is around 5.3 and the standard deviation is around 3.4.

```@example bayes
using AdvChemStat

p1 = histogram(m,label="mean",normalize=:probability)
xlabel!("Mean")
ylabel!("Probablity")

p2 = histogram(st,label="Standard deviation",normalize=:probability)
xlabel!("Standard deviation")
ylabel!("Probablity")

plot(p1,p2,layout=(2,1))


```

However, these results do not give us a more clear picture of the distribution of the measurements, suggesting that the conventional methods may not be adequate for these assessment. Now let's us Bayesian statistics to tackle this issue. As first step, we can assume a flat/uniform prior distribution. Based on this assumption, our simplified Bayes equation will become: 

```math 

P(A \mid B) \propto P(B \mid A) \times 1.

```

This set up of the Bayes Theorem results in so called [maximum likelihood estimate](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), which is similar to the outcome of the bootstrapping results. However, these results can be improved via incorporation of an informative prior distribution into our calculations. To do so we need to follow the below steps. 

#### Step 1 (prior selection): 

In this case, by looking at the distribution of the data we can consider a [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) as an adequate prior distribution for our data. Gamma distribution similar to normal distribution has two parameters: *k* shape and *``\theta``* scale. To estimate these parameters based on our measurements we need to fit this distribution to our data.   

```@example bayes
using AdvChemStat

pri = fit_mle(Gamma,x)

histogram(x,bins=5,label="Data",normalize=true)
plot!(fit_mle(Gamma, x),label="Model")
xlabel!("Concentration (ppb)")
ylabel!("Probablity")


```

#### Step 2 (assuming an average value):

In this step we assume that the true mean of or distribution is around 10 (this is only as example) and the mean actually has a normal distribution with a standard deviation same as *X*. By doing so, we can calculate ``P(A \mid B) = P(B \mid A) \times P(B)``, which is the probability of every measurement given the assumed average. The product of all these probabilities will result in the likelihood of 10 being the true mean of the distribution. Repeating this for all measured values, we will end up with the distribution of mean and its standard deviation. 

```@example bayes
using AdvChemStat

function uncertainty_estimate(x,pri)
    target = collect(range(minimum(x),maximum(x),length = 10000))       # Generate a set of potential mean values
    post = zeros(length(target))                                        # Generate the posterior distribution vector

    for i =1:length(target)

        dist = Normal(target[i],std(x))                                 # Distribution of the assumed mean
        tv = 1                                                          # Initialize the likelihood values
        for j = 1:length(x)

            tv = tv * pdf(dist,x[j]) * pdf(pri,x[j])                    # Updating the likelihood over each iteration

        end

        post[i] = tv


    end 

    post = post ./ sum(post)                                                # Normalize the posterior distribution

    return post, target
end 

post, target = uncertainty_estimate(x,pri)


```
The example of assumed mean values would look like this. 

```@example bayes
using AdvChemStat

histogram(x,bins=5,label="Data",normalize=:probability)
plot!(fit_mle(Gamma,x),label="Prior distribution")
plot!(Normal(10.5,std(x)),label="Potential distribution I")
plot!(Normal(8.5,std(x)),label="Potential distribution II")
plot!(Normal(3.5,std(x)),label="Potential distribution III")
#plot!(fit_mle(Normal, m),label="Model")
xlabel!("Concentration (ppb)")
ylabel!("Probability")

```

The final distribution on the other hand looks as below. Please note the resolution of this evaluation is highly dependent on the dataset. For example here we are generating a vector of 10000 members between 0 and 10, which may or may not be too fine. 

```@example bayes
using AdvChemStat

histogram(x,bins=5,label="Data",normalize=:probability)
plot!(fit_mle(Gamma,x),label="Prior distribution")
plot!(fit_mle(Normal, x),label="MLE")
plot!(target,10^3 .* post,label="Posterior distribution * 1000",c=:black)
#plot!(fit_mle(Normal, m),label="Model")
xlabel!("Concentration (ppb)")
ylabel!("Probablity")

```

As expected the predicted average based on the posterior probability distribution is very close to the one based on MLE. However, if we look at the uncertainty levels of this distribution (i.e. the ``\sigma``), we can see a much more accurate estimation of the distribution, which can be very important in different applications. 

### Bayesian regression

Bayesian statistics can also be used for solving regression problems. The [Bayesian regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) is an extension of the the uncertainty assessment, where the Bayes theorem is used to estimate the coefficients of a model starting from a flat prior for all the coefficients. The added step here is to use the sum square errors to improve the prior distribution of the coefficients. Typically these complex modeling problems are solved using a combination of Bayes theorem and Monte Carlo simulations 

## Additional Resources

You can find a lot of resources related to Bayesian Statistics and inference ([book](https://bayesiancomputationbook.com/welcome.html)), [julia package](https://turing.ml/v0.23/), and [python package](https://juanitorduz.github.io/intro_pymc3/). Additionally, the following videos are highly informative for better learning the Bayesian statistical concepts. 
* [MIT Open Courseware](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-17-video/)
* [Datacamp](https://www.youtube.com/watch?v=3OJEae7Qb_o&themeRefresh=1) 