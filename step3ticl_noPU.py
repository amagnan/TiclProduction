import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process('TICL',Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.MessageLogger = cms.Service("MessageLogger",
#                                       destinations = cms.untracked.vstring("debug"),
#                                     debugModules = cms.untracked.vstring("*"),
#                                     categories = cms.untracked.vstring("HGCPatternRecoByCA"),
#                                     debug = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
#                                                                        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
#                                                                        default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
#                                                                        HGCPatternRecoByCA = cms.untracked.PSet(limit = cms.untracked.int32(100000000))
#                                                                        )
#                                     )


# MessageLogger customizations
#process.MessageLogger.cerr.enable = False
#process.MessageLogger.cout.enable = False
#labels = ['SimTracks', 'SimVertices', 'GenParticles', 'TrackingParticles', 'CaloParticles', 'SimClusters']
#messageLogger = dict()
#for category in labels:
#    main_key = '%sMessageLogger'%(category)
#    category_key = 'CaloParticleDebugger%s'%(category)
#    messageLogger[main_key] = dict(
#            filename = '%s_%s.log' % (input_filename.replace('.root',''), category),
#            threshold = 'INFO',
#            default = dict(limit=0)
#            )
#    messageLogger[main_key][category_key] = dict(limit=-1)
#    # First create defaults
#    setattr(process.MessageLogger.files, category, dict())
#    # Then modify them
#    setattr(process.MessageLogger.files, category, messageLogger[main_key])


#process.MessageLogger = cms.Service(
#    "MessageLogger",
#    destinations       =  cms.untracked.vstring('debugmessages','cerr'),
#    #categories         = cms.untracked.vstring('*'),
#    debugModules       = cms.untracked.vstring('*'),
#    debugmessages      = cms.untracked.PSet(
#        threshold =  cms.untracked.string('DEBUG'),
#    ),
#    cerr               = cms.untracked.PSet(
#        threshold  = cms.untracked.string('INFO') 
#    )
#)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:/eos/user/s/shghosh/HGCALNTUPS/STEP3_PI.root'),
    fileNames = cms.untracked.vstring('file:step3.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
#process.source.skipEvents = cms.untracked.uint32(38)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    TryToContinue = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    holdsReferencesToDeleteEarly = cms.untracked.VPSet(),
    makeTriggerResults = cms.obsolete.untracked.bool,
    modulesToCallForTryToContinue = cms.untracked.vstring(),
    modulesToIgnoreForDeleteEarly = cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTTICLoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3ticl.root'),
    outputCommands = cms.untracked.vstring( (
        'drop *', 
        'keep *_HGCalRecHit_*_*', 
        'keep *_hgcalMergeLayerClusters_*_*', 
#        'keep *_hgcalLayerClusters_timeLayerCluster_*', 
#        'keep *_hgcalLayerClusters_InitialLayerClustersMask_*',
        'keep *_hgcalMultiClusters_*_*', 
        'keep *_iterHGCalMultiClusters_*_*',
        'keep *_ticlTracksters*_*_*', 
        'keep *_ticlSimTracksters*_*_*',
        'keep *_ticlMultiClustersFromTracksters*_*_*', 
        'keep *_ticlMultiClustersFromSimTracksters*_*_*', 
        'keep *SimTrack*_g4SimHits_*_SIM', 
        'keep *SimVertex*_g4SimHits_*_SIM', 
        'keep edmHepMCProduct_source_*_*', 
        'keep edmHepMCProduct_generatorSmeared_*_*', 
        'keep *_genParticle*_*_*', 
        'keep recoGenMETs_*_*_*', 
        'keep *_ak4GenJets_*_*', 
        'keep *_ak8GenJets_*_*', 
        'keep *_ak4GenJetsNoNu_*_*', 
        'keep *_ak8GenJetsNoNu_*_*', 
        'keep *_generator_*_*', 
        'keep *_mix*_MergedCaloTruth_*',
        'keep LHERunInfoProduct_*_*_*', 
        'keep LHEEventProduct_*_*_*', 
        'keep *_genPUProtons_*_*', 
    ) ),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

#process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T25', '')

#rerun default iterations
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

process.trackdnn_source = cms.ESSource(
    "EmptyESSource",
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('TfGraphRecord')
)                                                                                                                                                     
process.tracksterSelectionTf = cms.ESProducer(
    "TfGraphDefProducer",
    ComponentName = cms.string('tracksterSelectionTf'),
    FileName = cms.FileInPath('RecoHGCal/TICL/data/tf_models/energy_id_v0.pb'),
    appendToDataLabel = cms.string('')
)


#from RecoHGCal.TICL.TrkEMStep_cff import *
#from RecoHGCal.TICL.EMStep_cff import *
#from RecoHGCal.TICL.TrkStep_cff import *
#from RecoHGCal.TICL.HADStep_cff import *
#from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.CLUE3DHighStep_cff import *

#ticlTrackstersTrkEM.algo_verbosity = 5
#ticlTrackstersEM.algo_verbosity = 5
#ticlTrackstersTrk.algo_verbosity = 5
#ticlTrackstersHAD.algo_verbosity = 5

#make several TiCL iterations
#process.load('TiclProduction.Configuration.myDUMMYiterations_cff')
#process.load('TiclProduction.Configuration.myEMiterations_cff')
#process.load('TiclProduction.Configuration.myHADiterations_cff')
#process.load('TiclProduction.Configuration.myTRKiterations_cff')


#process.load('TiclProduction.Configuration.mySIMiterations_cff')

#cms.Path(
#HGCalUncalibRecHit,HGCalRecHit,hgcalRecHitMapProducer,
#hgcalLayerClusters,hgcalMultiClusters,particleFlowRecHitHGC,particleFlowClusterHGCal,particleFlowClusterHGCalFromMultiCl,
#particleFlowSuperClusterHGCal,particleFlowSuperClusterHGCalFromMultiCl,
#ticlLayerTileProducer,
#ticlSeedingGlobal,filteredLayerClustersDummy,ticlTrackstersDummy,ticlMultiClustersFromTrackstersDummy,
#ticlSeedingTrk,filteredLayerClustersTrkEM,ticlTrackstersTrkEM,ticlMultiClustersFromTrackstersTrkEM,
#ticlSeedingGlobal,filteredLayerClustersEM,ticlTrackstersEM,ticlMultiClustersFromTrackstersEM,
#ticlSeedingTrk,filteredLayerClustersTrk,ticlTrackstersTrk,ticlMultiClustersFromTrackstersTrk,
#ticlSeedingGlobal,filteredLayerClustersHAD,ticlTrackstersHAD,ticlMultiClustersFromTrackstersHAD,
#ticlTrackstersMerge,ticlMultiClustersFromTrackstersMerge)
#pfTICL,hgcalTrackCollection,
#)

#process.sim_path = cms.Sequence(process.sim_task)


from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import *
process.hgcalLayerClustersEE.plugin.doDensity=cms.bool(True)
process.hgcalLayerClustersHSi.plugin.doDensity=cms.bool(True)
process.hgcalLayerClustersHSci.plugin.doDensity=cms.bool(True)

process.make_new_lc = cms.Task(
    process.hgcalLayerClustersEE,
    process.hgcalLayerClustersHSi,
    process.hgcalLayerClustersHSci,
    process.hgcalMergeLayerClusters
    )

process.tile_task = cms.Task(process.trackdnn_source,process.ticlLayerTileProducer)

process.def_ticl = cms.Task(
    #process.ticlTrkEMStepTask,process.ticlEMStepTask,
    #process.ticlTrkStepTask,process.ticlHADStepTask
    process.ticlCLUE3DHighStepTask,
)


process.ticl_seq = cms.Sequence(
#    process.sim_task,
    process.make_new_lc,
    process.tile_task,
#    process.dummy_task,
#    process.em_task,
#    process.had_task,
#    process.trk_task,
    process.def_ticl,
)

#process.ticl_seq = cms.Sequence(process.ticl_task)

process.ticl_step = cms.Path(process.ticl_seq)

process.FEVTTICLoutput_step = cms.EndPath(process.FEVTTICLoutput)


# Schedule definition
process.schedule = cms.Schedule(
    process.ticl_step,
    process.FEVTTICLoutput_step)

# customisation of the process.

# customisation of the process.

# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
#from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
#process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
