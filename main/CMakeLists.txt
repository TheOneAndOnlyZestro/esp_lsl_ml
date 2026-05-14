idf_component_register(SRCS "main.cpp" "Microinterface/Model.cpp" "model_flash.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_timer spi_flash esp_driver_uart esp_partition)
