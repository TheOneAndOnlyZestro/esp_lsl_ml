idf_component_register(SRCS "lsl_handle.cpp" "main.cpp" "Microinterface/Model.cpp" "model_flash.cpp" "lsl_handle.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_timer spi_flash esp_driver_uart esp_partition esp_psram liblsl_esp32 esp-tflite-micro)

# Put our model on flash
esptool_py_flash_to_partition(flash "model" "${CMAKE_CURRENT_SOURCE_DIR}/EMG_TinyML_LiteRT_ConcatFix.tflite")