import FWCore.ParameterSet.Config as cms
from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal

filteredLayerClustersEM1 = cms.EDProducer("FilteredLayerClustersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    LayerClustersInputMask = cms.InputTag("hgcalLayerClusters","InitialLayerClustersMask"),
    algo_number = cms.int32(8),
    clusterFilter = cms.string('ClusterFilterByAlgoAndSizeAndLayerRange'),
    iteration_label = cms.string('EM1'),
    max_cluster_size = cms.int32(9999),
    max_layerId = cms.int32(30),
    mightGet = cms.optional.untracked.vstring,
    min_cluster_size = cms.int32(1),
    min_layerId = cms.int32(0)
)

filteredLayerClustersEM2 = filteredLayerClustersEM1.clone()
filteredLayerClustersEM2.min_cluster_size = cms.int32(2)
filteredLayerClustersEM2.iteration_label = cms.string('EM2')

filteredLayerClustersEM3 = filteredLayerClustersEM1.clone()
filteredLayerClustersEM3.min_cluster_size = cms.int32(3)
filteredLayerClustersEM3.iteration_label = cms.string('EM3')

ticlTrackstersEM1 = cms.EDProducer(
    "TrackstersProducer",
    detector = cms.string('HGCAL'),
    layer_clusters = cms.InputTag('hgcalLayerClusters'),
    filtered_mask = cms.InputTag('filteredLayerClustersEM1', 'EM1'),
    original_mask = cms.InputTag('hgcalLayerClusters', 'InitialLayerClustersMask'),
    time_layerclusters = cms.InputTag('hgcalLayerClusters', 'timeLayerCluster'),
    layer_clusters_tiles = cms.InputTag('ticlLayerTileProducer'),
    layer_clusters_hfnose_tiles = cms.InputTag('ticlLayerTileHFNose'),
    seeding_regions = cms.InputTag('ticlSeedingGlobal'),
    patternRecognitionBy = cms.string('CA'),
    itername = cms.string('EM1'),
    tfDnnLabel = cms.string('tracksterSelectionTf'),
    pluginPatternRecognitionByCA = cms.PSet(
        algo_verbosity = cms.int32(0),
        oneTracksterPerTrackSeed = cms.bool(False),
        promoteEmptyRegionToTrackster = cms.bool(False),
        out_in_dfs = cms.bool(True),
        max_out_in_hops = cms.int32(1),
        min_cos_theta = cms.double(0.97),
        min_cos_pointing = cms.double(0.90),
        root_doublet_max_distance_from_seed_squared = cms.double(9999),
        etaLimitIncreaseWindow = cms.double(2.1),
        skip_layers = cms.int32(2),
        max_missing_layers_in_trackster = cms.int32(1),
        shower_start_max_layer = cms.int32(9999),
        min_layers_per_trackster = cms.int32(5),
        filter_on_categories = cms.vint32(0,1),
        pid_threshold = cms.double(0),
        energy_em_over_total_threshold = cms.double(-1),
        max_longitudinal_sigmaPCA = cms.double(9999),
        max_delta_time = cms.double(3),
        eid_input_name = cms.string('input'),
        eid_output_name_energy = cms.string('output/regressed_energy'),
        eid_output_name_id = cms.string('output/id_probabilities'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        siblings_maxRSquared = cms.vdouble(
            0.0006,
            0.0006,
            0.0006
        ),
        type = cms.string('CA')
        
    ),
    pluginPatternRecognitionByCLUE3D = cms.PSet(
        algo_verbosity = cms.int32(0),
        criticalDensity = cms.double(4),
        densitySiblingLayers = cms.int32(3),
        densityEtaPhiDistanceSqr = cms.double(0.0008),
        densityOnSameLayer = cms.bool(False),
        criticalEtaPhiDistance = cms.double(0.035),
        outlierMultiplier = cms.double(2),
        minNumLayerCluster = cms.int32(2),
        eid_input_name = cms.string('input'),
        eid_output_name_energy = cms.string('output/regressed_energy'),
        eid_output_name_id = cms.string('output/id_probabilities'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        type = cms.string('CLUE3D')
        
    ),
    pluginPatternRecognitionByFastJet = cms.PSet(
        algo_verbosity = cms.int32(0),
        antikt_radius = cms.double(0.09),
        minNumLayerCluster = cms.int32(5),
        eid_input_name = cms.string('input'),
        eid_output_name_energy = cms.string('output/regressed_energy'),
        eid_output_name_id = cms.string('output/id_probabilities'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_layers = cms.int32(50),
        eid_n_clusters = cms.int32(10),
        type = cms.string('FastJet')
        
    ),
    mightGet = cms.optional.untracked.vstring
)


ticlTrackstersEM2 = ticlTrackstersEM1.clone()
ticlTrackstersEM2.filtered_mask = cms.InputTag("filteredLayerClustersEM2","EM2")
ticlTrackstersEM2.itername = cms.string('EM2')

ticlTrackstersEM3 = ticlTrackstersEM1.clone()
ticlTrackstersEM3.filtered_mask = cms.InputTag("filteredLayerClustersEM3","EM3")
ticlTrackstersEM3.itername = cms.string('EM3')

ticlTrackstersCLUE3D3 = ticlTrackstersEM1.clone()
ticlTrackstersCLUE3D3.filtered_mask = cms.InputTag("filteredLayerClustersEM3","EM3")
ticlTrackstersCLUE3D3.itername = cms.string('CLUE3D3')
ticlTrackstersCLUE3D3.patternRecognitionBy = cms.string('CLUE3D')

em_task =  cms.Task(
    ticlSeedingGlobal,filteredLayerClustersEM1,ticlTrackstersEM1,
    ticlSeedingGlobal,filteredLayerClustersEM2,ticlTrackstersEM2,
    ticlSeedingGlobal,filteredLayerClustersEM3,ticlTrackstersEM3,
    ticlTrackstersCLUE3D3
)
