using Cairn, Molly
using Unitful
using StaticArrays
using SpecialPolynomials
using Test


@testset "System and Data Tests" begin


    xtrain = [[0.8464987622491895, -0.7801390420313014],
    [0.096079218176701, -0.9623723102484034],
    [0.794900754980532, -0.0790635268608757],
    [0.2997659303869744, -0.2798944543828574],
    [0.3921279139032157, -0.1410288797166183]]
    conftrain = [SVector{2}(x) .* u"nm" for x in xtrain] 
    ntrain = length(xtrain)

    ref = MullerBrown()
    pce = PolynomialChaos(3, 2, Jacobi{0.5,0.5})
    pce.params = ones(length(pce.basis))

    

    ## test define_sys and define_ens
    sysa = define_sys(pce, xtrain[1],
        data=Dict(
            "energy_descriptors" => Float64[],
            "force_descriptors" => Vector[],
        )
    ) # single atom system
    sysb = define_ens(pce, xtrain) # ensemble of single-atom systems
    sysc = define_sys(pce, xtrain) # multi-atom system
    
    sysd = define_sys(pce, conftrain[1]) # single atom system
    syse = define_ens(pce, conftrain) # ensemble of single-atom systems
    sysf = define_sys(pce, conftrain) # multi-atom system
    
    sysg = define_sys(pce, xtrain[1], 
        loggers=(coords=CoordinateLogger(10; dims=2),),
    ) # with logger
    sysh = remove_loggers(sysg) # without logger

    @test length(sysa.coords) == 1
    @test length(sysb[1].coords) == 1
    @test length(sysb) == ntrain
    @test length(sysc.coords) == ntrain
    @test typeof(sysa.coords[1]) <: SVector

    @test sysd.coords == sysa.coords
    @test syse[1].coords == sysb[1].coords
    @test sysf.coords == sysc.coords

    @test typeof(sysg.loggers) <: NamedTuple
    @test typeof(sysh.loggers) == Tuple{}



    ## test ConfigurationData quantities 
    e = potential_energy(sysa, pce)
    edata = Energy(e)
    f = forces(sysa, pce)
    fdata = Forces(f)
    ed = compute_local_descriptors(sysa, pce) # populates sysa.data
    edescr = LocalDescriptors(ed)
    fd = compute_force_descriptors(sysa, pce) # populates sysa.data
    fdescr = ForceDescriptors(fd)

    c = Configuration(edata, fdata, edescr, fdescr)

    @test sysa.data["energy_descriptors"] != Float64[]
    @test sysa.data["force_descriptors"] != Vector[]

    @test typeof(e) <: Quantity
    @test get_values(e) == edata.d
    @test unit(e) == edata.u
    @test typeof(f[1]) <: SVector{2, <:Quantity}
    @test get_values(f[1]) == fdata.f[1].f
    @test unit(f[1][1]) == fdata.f[1].u

    @test ed == get_values(edescr)
    @test fd == get_values(fdescr)
    @test typeof(edescr) <: LocalDescriptors
    @test typeof(edescr[1]) <: LocalDescriptor
    @test typeof(fdescr) <: ForceDescriptors
    @test typeof(fdescr[1]) <: ForceDescriptor

    @test c.data[Energy].d == get_values(e)
    @test c.data[Forces].f[1].f == get_values(f[1])
    @test get_values(c.data[LocalDescriptors][1]) == ed[1]
    @test get_values(c.data[ForceDescriptors][1]) == fd[1]



    ## test DataSet construction
    conf = TrainConfiguration(sysa, ref) 
    ds = TrainDataSet(sysb, ref)

    @test typeof(conf) <: Configuration
    @test typeof(ds) <: DataSet 
    @test conf.data[Energy] == ds[1].data[Energy]

end