module AdvChemStat

using LinearAlgebra
using CSV
using PyCall
using DataFrames 
using RDatasets
using Plots
using Random
using Statistics
using ScikitLearnBase
using ScikitLearn 
using ScikitLearn.Skcore: @sk_import
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using Clustering
import StatsPlots as sp
using Conda
using Distributions
using Distances 
using StatsBase
using HDF5
using JLD


include("ACS_Extra_Fun.jl")
include("Curve_Res.jl")

export dataset, plot, plot!, scatter, scatter!, xlabel!, ylabel!, title!, @sk_import, eigvals, eigvecs, transpose, svd, diagm,
pinv, diag, xlims!, ylims!, describe, mean, median, bar, bar!, diagind, hclust, sp, pairwise, kmeans, assignments, annotate!, histogram,
histogram!, Normal, Gamma, fit_mle, std, pdf, cdf, read_ACS_data, cross_val_score, train_test_split, sample, CSV, DataFrame, cov, cor, Diagonal, I, inv, 
heatmap, heatmap!, dist_calc, NMF, SIMPLE, FNNLS, UnimodalFixedUpdate, UnimodalUpdate, UnimodalLeastSquares, MonotoneRegression, MCR_ALS_jw_March29
# Write your package code here.




end
