idf_component_register(
SRCS 
    "master_handle.cpp" "wifi_helper.c"
    "main.cpp" "Microinterface/Model.cpp" "model_flash.cpp"
    "lsl_handle.cpp"

INCLUDE_DIRS "."

REQUIRES driver esp_timer spi_flash 
esp_driver_uart esp_partition esp_psram 
liblsl_esp32 esp-tflite-micro nvs_flash
esp_wifi esp_event
)

# Put our model on flash
esptool_py_flash_to_partition(flash "model_1" "${CMAKE_CURRENT_SOURCE_DIR}/First_EMG_TinyML_LiteRT_ConcatFix.tflite")
esptool_py_flash_to_partition(flash "model_2" "${CMAKE_CURRENT_SOURCE_DIR}/Second_EMG_TinyML_LiteRT_ConcatFix.tflite")