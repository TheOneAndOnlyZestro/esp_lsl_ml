idf_component_register(
SRCS 
    "master_handle.cpp" "wifi_helper.c"
    "main.cpp" "Microinterface/Model.cpp" "model_flash.cpp"
    "benchmark_handle.cpp"
    "lsl_handle.cpp"
    "OptimizedNativeLSTM.cpp"

INCLUDE_DIRS "." "./binary_manifests"

REQUIRES driver esp_timer spi_flash 
    esp_driver_uart esp_partition esp_psram 
    liblsl_esp32 esp-tflite-micro nvs_flash
    esp_wifi esp_event espressif__esp-nn
)

# Put our model on flash
# esptool_py_flash_to_partition(flash "model_1" "${CMAKE_CURRENT_SOURCE_DIR}/First_EMG_TinyML_LiteRT_ConcatFix_w500_int8.tflite")
# esptool_py_flash_to_partition(flash "model_2" "${CMAKE_CURRENT_SOURCE_DIR}/Second_EMG_TinyML_LiteRT_ConcatFix_w500_int8.tflite")

# esptool_py_flash_to_partition(flash "models" "${CMAKE_CURRENT_SOURCE_DIR}/models.bin")

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/dense_for_espnn/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/dense_for_espnn/brute_data_0.bin")

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/model_all/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/model_all/data.bin")

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/four_models_corrected/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/four_models_corrected/data.bin")

#esptool_py_flash_to_partition(flash "lstm_weights" "${CMAKE_CURRENT_SOURCE_DIR}/lstm_nn_binaries/c16/lstm_q.bin")

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/four_models_old_model/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/four_models_old_model/data.bin")

# esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/dense_layers_seed1/models_0.bin")
# esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/dense_layers_seed1/brute_data_0.bin")

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/four_models_step_500/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/four_models_step_500/data.bin")

# esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/calibrated_better/models_0.bin")
# esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/better_trials/trial_15/data.bin")

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/conv-40/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/conv-40/data.bin")


# esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/lstm-hiddensize-verification2/models_0.bin")
# esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/lstm-hiddensize-verification2/data.bin")

# esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/lstm-layers-verification/models_0.bin")
# esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/lstm-layers-verification/data.bin")

# esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/Conv2D-InputSize-Verification/models_0.bin")
# esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/Conv2D-InputSize-Verification/data.bin")

# esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/refactored_models_verification/models_0.bin")
# esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/refactored_models_verification/data.bin")

esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/quant_fidelity_test_verification/models_0.bin")
esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/quant_fidelity_test_verification/data.bin")