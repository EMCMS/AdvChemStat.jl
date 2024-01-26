using CSV 
using DataFrames
using HTTP 



######################################################
# Function

######################################################
# Reading datasets from the internal repo

function read_ACS_data(file_name)
    url = "https://raw.githubusercontent.com/EMCMS/AdvChemStat.jl/main/datasets"
    
    
    data = DataFrame(CSV.File(HTTP.get(url * "/" *  file_name).body))
    return data
end


######################################################
# Building the euclidian distance matrix 


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


#file_name = "mtcars.csv"

######################################################
# Random integers between two numbers 

# a = 1 
# b = 100
# n = 100 

function randi(a,b,n)

    if a > b

        out = Int.(round.(b .+ (a - b) .* rand(n)))
    elseif a < b
        out = Int.(round.(a .+ (b - a) .* rand(n)))
    else 
        println("a and b should be different!")
        out = []

    end 

    return(out)

end 
