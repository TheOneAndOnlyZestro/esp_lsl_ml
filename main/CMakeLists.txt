idf_component_register(SRCS "main.cpp" "Microinterface/Model.cpp" "model_flash.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_timer spi_flash esp_driver_uart esp_partition esp_psram)

# Put our model on flash
esptool_py_flash_to_partition(flash "model" "${CMAKE_CURRENT_SOURCE_DIR}/EMG_TinyML_float32.tflite")