import Pkg
Pkg.activate(".")
Pkg.instantiate()

# do all adds
Pkg.add("JLD2")
# Pkg.add("TensorKit")
# Pkg.add("MPSKit")
# Pkg.add("MPSKitModels")
# Pkg.add("Plots")

# println("Libraries added")


#Pkg.add(; name="TensorKit", rev="master")

#Pkg.develop(; path="/kyukon/scratch/gent/444/vsc44475/GQ")
