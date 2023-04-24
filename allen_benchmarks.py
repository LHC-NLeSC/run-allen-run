###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
# Throughput test 12/08/2020
benchmark_weights = dict(
    [
        # Manual entries
        # 'initialize_event_lists' is added automatically at the beginning of the sequence always
        ("initialize_number_of_events", 1),
        ("mep_layout", 2),
        ("populate_odin_banks", 3),
        ("odin_error_filter", 4),
        ("Hlt1Passthrough", 5),
        ("Hlt1NoBeam", 6),
        ("Hlt1BeamOne", 7),
        ("Hlt1BeamTwo", 8),
        ("Hlt1BothBeams", 9),
        ("Hlt1ODINLumi", 10),
        ("Hlt1ODINVeloOpen", 11),
        ("Hlt1ODINNoBias", 12),
        ("host_scifi_banks", 13),
        ("scifi_gec", 14),
        ("odin_error_filter_AND_scifi_gec", 15),
        ("host_ut_banks", 16),
        ("ut_gec", 17),
        ("odin_error_filter_AND_scifi_gec_AND_ut_gec", 18),
        ("Hlt1GECPassthrough", 19),
        ("ut_banks", 20),
        ("ut_calculate_number_of_hits", 21),
        ("calo_count_digits", 22),
        ("muon_banks", 23),
        ("muon_calculate_srq_size", 24),
        ("velo_banks", 25),
        ("ecal_banks", 26),
        ("calculate_number_of_retinaclusters_each_sensor_pair", 27),
        ("scifi_banks", 28),
        ("scifi_calculate_cluster_count", 29),
        ("prefix_sum_ut_hits", 30),
        ("ut_pre_decode", 31),
        ("ut_find_permutation", 32),
        ("ut_decode_raw_banks_in_order", 33),
        ("prefix_sum_scifi_hits", 34),
        ("scifi_pre_decode", 35),
        ("scifi_raw_bank_decoder", 36),
        ("muon_srq_prefix_sum", 37),
        ("muon_populate_tile_and_tdc", 38),
        ("muon_station_ocurrence_prefix_sum", 39),
        ("muon_add_coords_crossing_maps", 40),
        ("muon_populate_hits", 41),
        ("find_muon_hits", 42),
        ("prefix_sum_muon_tracks_find_hits", 43),
        ("consolidate_muon_t", 44),
        ("Hlt1OneMuonTrackLine", 45),
        ("prefix_sum_offsets_estimated_input_size", 46),
        ("decode_retinaclusters", 47),
        ("velo_search_by_triplet", 48),
        ("velo_three_hit_tracks_filter", 49),
        ("prefix_sum_offsets_velo_tracks", 50),
        ("prefix_sum_ecal_num_digits", 51),
        ("calo_decode", 52),
        ("calo_seed_clusters", 53),
        ("prefix_sum_ecal_num_clusters", 54),
        ("calo_find_clusters", 55),
        ("filter_clusters", 56),
        ("prefix_sum_filtered_calo_clusters", 57),
        ("calo_find_twoclusters", 58),
        ("Hlt1Bs2GammaGamma", 59),
        ("prefix_sum_offsets_number_of_three_hit_tracks_filtered", 60),
        ("velo_copy_track_hit_number", 61),
        ("prefix_sum_offsets_velo_track_hit_number", 62),
        ("Hlt1VeloMicroBias", 63),
        ("velo_consolidate_tracks", 64),
        ("velo_kalman_filter", 65),
        ("Hlt1BeamGas", 66),
        ("ut_select_velo_tracks", 67),
        ("ut_search_windows", 68),
        ("ut_select_velo_tracks_with_windows", 69),
        ("compass_ut", 70),
        ("brem_recovery", 71),
        ("Hlt1SingleHighEt", 72),
        ("pv_beamline_extrapolate", 73),
        ("pv_beamline_histo", 74),
        ("pv_beamline_peak", 75),
        ("pv_beamline_calculate_denom", 76),
        ("pv_beamline_multi_fitter", 77),
        ("pv_beamline_cleanup", 78),
        ("prefix_sum_ut_tracks", 79),
        ("ut_copy_track_hit_number", 80),
        ("prefix_sum_ut_track_hit_number", 81),
        ("ut_consolidate_tracks", 82),
        ("get_type_id", 83),
        ("lf_search_initial_windows", 84),
        ("lf_triplet_seeding", 85),
        ("lf_create_tracks", 86),
        ("lf_quality_filter_length", 87),
        ("lf_quality_filter", 88),
        ("prefix_sum_forward_tracks", 89),
        ("scifi_copy_track_hit_number", 90),
        ("prefix_sum_scifi_track_hit_number", 91),
        ("scifi_consolidate_tracks", 92),
        ("track_digit_selective_matching", 93),
        ("is_muon", 94),
        ("kalman_velo_only", 95),
        ("make_lepton_id", 96),
        ("make_long_track_particles", 97),
        ("Hlt1TrackMVA", 98),
        ("Hlt1SingleHighPtMuon", 99),
        ("Hlt1SingleHighPtMuonNoMuID", 100),
        ("Hlt1LowPtMuon", 101),
        ("Hlt1TrackMuonMVA", 102),
        ("Hlt1RICH1Alignment", 103),
        ("Hlt1RICH2Alignment", 104),
        ("filter_tracks", 105),
        ("momentum_brem_correction", 106),
        ("Hlt1TrackElectronMVA", 107),
        ("Hlt1SingleHighPtElectron", 108),
        ("Hlt1DisplacedLeptons", 109),
        ("prefix_sum_secondary_vertices", 110),
        ("fit_secondary_vertices", 111),
        ("Hlt1KsToPiPi", 112),
        ("Hlt1TwoTrackKs", 113),
        ("Hlt1D2KK", 114),
        ("Hlt1D2KPi", 115),
        ("Hlt1D2PiPi", 116),
        ("Hlt1DiMuonHighMass", 117),
        ("Hlt1DiMuonLowMass", 118),
        ("Hlt1DiMuonSoft", 119),
        ("Hlt1LowPtDiMuon", 120),
        ("Hlt1DiMuonNoIP", 121),
        ("Hlt1DiMuonNoIP_ss", 122),
        ("Hlt1DisplacedDielectron", 123),
        ("Hlt1D2KPiAlignment", 124),
        ("Hlt1DiMuonHighMassAlignment", 125),
        ("Hlt1DisplacedDiMuonAlignment", 126),
        ("two_track_mva_evaluator", 127),
        ("Hlt1TwoTrackMVACharmXSec", 128),
        ("Hlt1TwoTrackMVA", 129),
        ("calc_max_combos", 130),
        ("prompt_vertex_evaluator", 131),
        ("Hlt1LowMassNoipDielectron_massSlice1_prompt", 132),
        ("Hlt1LowMassNoipDielectron_SS_massSlice1_prompt", 133),
        ("Hlt1LowMassNoipDielectron_massSlice2_prompt", 134),
        ("Hlt1LowMassNoipDielectron_SS_massSlice2_prompt", 135),
        ("Hlt1LowMassNoipDielectron_massSlice3_prompt", 136),
        ("Hlt1LowMassNoipDielectron_SS_massSlice3_prompt", 137),
        ("Hlt1LowMassNoipDielectron_massSlice4_prompt", 138),
        ("Hlt1LowMassNoipDielectron_SS_massSlice4_prompt", 139),
        ("Hlt1LowMassNoipDielectron_massSlice1_displaced", 140),
        ("Hlt1LowMassNoipDielectron_SS_massSlice1_displaced", 141),
        ("Hlt1LowMassNoipDielectron_massSlice2_displaced", 142),
        ("Hlt1LowMassNoipDielectron_SS_massSlice2_displaced", 143),
        ("Hlt1LowMassNoipDielectron_massSlice3_displaced", 144),
        ("Hlt1LowMassNoipDielectron_SS_massSlice3_displaced", 145),
        ("Hlt1LowMassNoipDielectron_massSlice4_displaced", 146),
        ("Hlt1LowMassNoipDielectron_SS_massSlice4_displaced", 147),
        ("prefix_sum_max_combos", 148),
        ("filter_svs", 149),
        ("prefix_sum_sv_combos", 150),
        ("svs_pair_candidate", 151),
        ("Hlt1TwoKs", 152),
        ("gather_selections", 153),
        ("calc_lumi_sum_size", 154),
        ("dec_reporter", 155),
        ("global_decision", 156),
        ("host_routingbits_writer", 157),
        ("rate_validator", 158),
        ("prefix_sum_lumi_size", 159),
        ("velo_total_tracks", 160),
        ("pv_lumi_counters", 161),
        ("scifi_lumi_counters", 162),
        ("muon_lumi_counters", 163),
        ("calo_lumi_counters", 164),
        ("make_lumi_summary", 165),
        ("prefix_sum_max_objects", 166),
        ("make_selected_object_lists", 167),
        ("prefix_sum_stdinfo_size", 168),
        ("prefix_sum_objtyp_size", 169),
        ("prefix_sum_candidate_count", 170),
        ("prefix_sum_hits_size", 171),
        ("prefix_sum_substr_size", 172),
        ("make_subbanks", 173),
        ("prefix_sum_selrep_size", 174),
        ("make_selreps", 175),
        ("GhostProbabilityNN0", 200),
        ("GhostProbabilityNN1", 200),
        ("GhostProbabilityNN2", 200),
        ("GhostProbabilityNN3", 200),
        ("GhostProbabilityNN4", 200),
        ("GhostProbabilityNN_HC", 200),
    ]
)

# Hardcoded on 29/07/2020
benchmark_efficiencies = dict([("gec", 0.9)])
