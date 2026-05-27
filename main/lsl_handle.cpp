#include "lsl_handle.h"

LSLHandle::LSLHandle() :
 m_inlet(nullptr), m_inlet_info(nullptr), m_outlet(nullptr), m_outlet_info(nullptr) {
    init_wifi();

    //init_outlet();
    init_inlet();
 }

 LSLHandle::~LSLHandle() {
    if (m_inlet) {
        lsl_esp32_destroy_inlet(m_inlet);
    }
    if (m_inlet_info) {
        lsl_esp32_destroy_streaminfo(m_inlet_info);
    }
    if (m_outlet) {
        lsl_esp32_destroy_outlet(m_outlet);
    }
    if (m_outlet_info) {
        lsl_esp32_destroy_streaminfo(m_outlet_info);
    }
 }

void LSLHandle::init_wifi() {
    /* Initialize NVS (required for WiFi) */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Connect to WiFi */
    ESP_LOGI("LSLHandle", "Connecting to WiFi...");
    if (wifi_helper_init_sta() != ESP_OK) {
        ESP_LOGE("LSLHandle", "WiFi connection failed. Cannot start LSL inlet.");
        return;
    }
}

const char* LSLHandle::get_inlet_info() const {
    if (!m_inlet_info) return nullptr;
    return lsl_esp32_get_name(m_inlet_info);
}

const char* LSLHandle::get_outlet_info() const {
    if (!m_outlet_info) return nullptr;
    return lsl_esp32_get_name(m_outlet_info);
}

void LSLHandle::init_inlet() {

    // Create Inlet first for receiving EMG data from Source
    ESP_LOGI("LSLHandle", "Resolving input stream '%s'...", CONFIG_LSL_TARGET_STREAM);

    int found = lsl_esp32_resolve_stream("name", CONFIG_LSL_TARGET_STREAM, 1.0, &m_inlet_info);
    if (!found || !m_inlet_info) {
        ESP_LOGE("LSLHandle", "Stream '%s' not found within 10 seconds", CONFIG_LSL_TARGET_STREAM);
        return;
    }

    int ch_count = lsl_esp32_get_channel_count(m_inlet_info);
    ESP_LOGI("LSLHandle", "Found stream: %s (%s, %dch @ %.0fHz)", lsl_esp32_get_name(m_inlet_info),
             lsl_esp32_get_type(m_inlet_info), ch_count, lsl_esp32_get_nominal_srate(m_inlet_info));


    ESP_LOGI("LSLHandle", "Creating inlet...");
    m_inlet = lsl_esp32_create_inlet(m_inlet_info);
    if (!m_inlet) {
        ESP_LOGE("LSLHandle", "Failed to create inlet");
        lsl_esp32_destroy_streaminfo(m_inlet_info);
        return;
    }

}
void LSLHandle::init_outlet() {

    // Create outlet for sending inference results back to Sink
    m_outlet_info = lsl_esp32_create_streaminfo(
        CONFIG_LSL_OUTLET_NAME, "EMG", CONFIG_OUTPUT_CHANNELS, CONFIG_OUTPUT_SAMPLE_RATE, LSL_ESP32_FMT_FLOAT32, "esp32_outlet_1");
    if (!m_outlet_info) {
        ESP_LOGE("LSLHandle", "Failed to create stream info");
        return;
    }

    m_outlet = lsl_esp32_create_outlet(m_outlet_info, 0, 360);
    if (!m_outlet) {
        ESP_LOGE("LSLHandle", "Failed to create outlet");
        lsl_esp32_destroy_streaminfo(m_outlet_info);
        return;
    }

}
lsl_esp32_err_t LSLHandle::pull_samples(float (&buffer)[CONFIG_INPUT_CHANNELS]) {
    if (!m_inlet) {
        return LSL_ESP32_ERR_INVALID_ARG;
    }

    lsl_esp32_err_t err = lsl_esp32_inlet_pull_sample_f(
        m_inlet, buffer, sizeof(buffer), &m_inlet_last_timestamp, 0.0);

    return err;
}

lsl_esp32_err_t LSLHandle::push_samples(const float (&buffer)[CONFIG_OUTPUT_CHANNELS]) {
    if (!m_outlet) {
        return LSL_ESP32_ERR_INVALID_ARG;
    }

    init_timestep(); 
    lsl_esp32_err_t err = lsl_esp32_push_sample_f(
        m_outlet, buffer, m_outlet_last_timestamp);

    return err;
}

void LSLHandle::init_timestep() {
    // Implementation for initializing timestep
    m_outlet_last_timestamp = lsl_esp32_local_clock();
}