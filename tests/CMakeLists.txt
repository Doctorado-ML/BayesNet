if(ENABLE_TESTING)
    include_directories(
        ${BayesNet_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/configured_files/include
    )
    file(GLOB_RECURSE BayesNet_SOURCES "${bayesnet_SOURCE_DIR}/bayesnet/*.cc")
    add_executable(TestBayesNet TestBayesNetwork.cc TestBayesNode.cc TestBayesClassifier.cc TestXSPnDE.cc TestXBA2DE.cc 
        TestBayesModels.cc TestBayesMetrics.cc TestFeatureSelection.cc TestBoostAODE.cc TestXBAODE.cc TestA2DE.cc 
        TestUtils.cc TestBayesEnsemble.cc TestModulesVersions.cc TestBoostA2DE.cc TestMST.cc TestXSPODE.cc ${BayesNet_SOURCES})
    target_link_libraries(TestBayesNet PUBLIC "${TORCH_LIBRARIES}" fimdlp::fimdlp PRIVATE Catch2::Catch2WithMain)
    add_test(NAME BayesNetworkTest COMMAND TestBayesNet)
    add_test(NAME A2DE COMMAND TestBayesNet "[A2DE]")
    add_test(NAME BoostA2DE COMMAND TestBayesNet "[BoostA2DE]")
    add_test(NAME BoostAODE COMMAND TestBayesNet "[BoostAODE]")
    add_test(NAME XSPODE COMMAND TestBayesNet "[XSPODE]")
    add_test(NAME XSPnDE COMMAND TestBayesNet "[XSPnDE]")
    add_test(NAME XBAODE COMMAND TestBayesNet "[XBAODE]")
    add_test(NAME XBA2DE COMMAND TestBayesNet "[XBA2DE]")
    add_test(NAME Classifier COMMAND TestBayesNet "[Classifier]")
    add_test(NAME Ensemble COMMAND TestBayesNet "[Ensemble]")
    add_test(NAME FeatureSelection COMMAND TestBayesNet "[FeatureSelection]")
    add_test(NAME Metrics COMMAND TestBayesNet "[Metrics]")
    add_test(NAME Models COMMAND TestBayesNet "[Models]")
    add_test(NAME Modules COMMAND TestBayesNet "[Modules]")
    add_test(NAME Network COMMAND TestBayesNet "[Network]")
    add_test(NAME Node COMMAND TestBayesNet "[Node]")
    add_test(NAME MST COMMAND TestBayesNet "[MST]")
endif(ENABLE_TESTING)
