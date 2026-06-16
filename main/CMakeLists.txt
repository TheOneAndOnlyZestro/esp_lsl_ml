idf_component_register(
SRCS 
    "master_handle.cpp" "wifi_helper.c"
    "main.cpp" "Microinterface/Model.cpp" "Microinterface/SequentialModel.cpp" "model_flash.cpp"
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

#esptool_py_flash_to_partition(flash "benchmark_models" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/tri_models/models_0.bin")
#esptool_py_flash_to_partition(flash "benchmark_data" "${CMAKE_CURRENT_SOURCE_DIR}/data_binaries/tri_models/data.bin")