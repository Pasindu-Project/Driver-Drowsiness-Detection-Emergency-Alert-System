#include <test2_inferencing.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_LPS22HB.h>
#include <Arduino_HS300x.h>
#include <Arduino_APDS9960.h>

#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f
#define N_SENSORS 18

#define BUZZER_PIN  D6
#define BUTTON_PIN  D5
#define PHONE_NUMBER "+94741696132" // Replace with your phone number

enum sensor_status {
    NOT_USED = -1,
    NOT_INIT,
    INIT,
    SAMPLED
};

typedef struct {
    const char *name;
    float *value;
    uint8_t (*poll_sensor)(void);
    bool (*init_sensor)(void);
    sensor_status status;
} eiSensors;

float ei_get_sign(float number);
bool init_IMU(void), init_HTS(void), init_BARO(void), init_APDS(void);
uint8_t poll_acc(void), poll_gyr(void), poll_mag(void), poll_HTS(void);
uint8_t poll_BARO(void), poll_APDS_color(void), poll_APDS_proximity(void), poll_APDS_gesture(void);
bool ei_connect_fusion_list(const char *input_list);
int8_t ei_find_axis(char *axis_name);

static const bool debug_nn = false;
static float data[N_SENSORS];
int8_t fusion_sensors[N_SENSORS];
int fusion_ix = 0;

bool buzzer_on = false;
bool call_in_progress = false;

eiSensors sensors[] = {
    "accX", &data[0], &poll_acc, &init_IMU, NOT_USED,
    "accY", &data[1], &poll_acc, &init_IMU, NOT_USED,
    "accZ", &data[2], &poll_acc, &init_IMU, NOT_USED,
    "gyrX", &data[3], &poll_gyr, &init_IMU, NOT_USED,
    "gyrY", &data[4], &poll_gyr, &init_IMU, NOT_USED,
    "gyrZ", &data[5], &poll_gyr, &init_IMU, NOT_USED,
    "magX", &data[6], &poll_mag, &init_IMU, NOT_USED,
    "magY", &data[7], &poll_mag, &init_IMU, NOT_USED,
    "magZ", &data[8], &poll_mag, &init_IMU, NOT_USED,
    "temperature", &data[9], &poll_HTS, &init_HTS, NOT_USED,
    "humidity", &data[10], &poll_HTS, &init_HTS, NOT_USED,
    "pressure", &data[11], &poll_BARO, &init_BARO, NOT_USED,
    "red", &data[12], &poll_APDS_color, &init_APDS, NOT_USED,
    "green", &data[13], &poll_APDS_color, &init_APDS, NOT_USED,
    "blue", &data[14], &poll_APDS_color, &init_APDS, NOT_USED,
    "brightness", &data[15], &poll_APDS_color, &init_APDS, NOT_USED,
    "proximity", &data[16], &poll_APDS_proximity, &init_APDS, NOT_USED,
    "gesture", &data[17], &poll_APDS_gesture, &init_APDS, NOT_USED,
};

// GSM call functions
void make_gsm_call(const char *number) {
    Serial1.println("AT"); delay(500);
    Serial1.print("ATD");
    Serial1.print(number);
    Serial1.println(";");
    delay(1000);
}

void end_gsm_call() {
    Serial1.println("ATH"); // Hang up
    delay(1000);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Edge Impulse Sensor Fusion Inference\r\n");

    if (!ei_connect_fusion_list(EI_CLASSIFIER_FUSION_AXES_STRING)) {
        ei_printf("ERR: Errors in sensor list detected\r\n");
        return;
    }

    for (int i = 0; i < fusion_ix; i++) {
        if (sensors[fusion_sensors[i]].status == NOT_INIT) {
            sensors[fusion_sensors[i]].status = (sensor_status)sensors[fusion_sensors[i]].init_sensor();
            ei_printf("%s axis sensor initialization %s\r\n",
                      sensors[fusion_sensors[i]].name,
                      sensors[fusion_sensors[i]].status ? "successful" : "failed");
        }
    }

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    Serial1.begin(9600); // SIM900A default
    delay(2000);
    Serial1.println("AT");
    delay(500);
    Serial1.println("AT+CLIP=1");
    delay(500);
}

void loop() {
    ei_printf("\nStarting inferencing in 2 seconds...\r\n");
    delay(500);

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != fusion_ix) {
        ei_printf("ERR: Sensors don't match the model\r\n");
        return;
    }

    ei_printf("Sampling...\r\n");
    float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
        int64_t next_tick = (int64_t)micros() + ((int64_t)EI_CLASSIFIER_INTERVAL_MS * 1000);

        for (int i = 0; i < fusion_ix; i++) {
            if (sensors[fusion_sensors[i]].status == INIT) {
                sensors[fusion_sensors[i]].poll_sensor();
                sensors[fusion_sensors[i]].status = SAMPLED;
            }
            if (sensors[fusion_sensors[i]].status == SAMPLED) {
                buffer[ix + i] = *sensors[fusion_sensors[i]].value;
                sensors[fusion_sensors[i]].status = INIT;
            }
        }

        int64_t wait_time = next_tick - (int64_t)micros();
        if (wait_time > 0) delayMicroseconds(wait_time);
    }

    signal_t signal;
    if (numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal) != 0) {
        ei_printf("ERR: Failed to create signal from buffer\n");
        return;
    }

    ei_impulse_result_t result = { 0 };
    if (run_classifier(&signal, &result, debug_nn) != EI_IMPULSE_OK) {
        ei_printf("ERR: Classifier failed\n");
        return;
    }

    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.):\r\n",
              result.timing.dsp, result.timing.classification, result.timing.anomaly);

    bool down_detected = false;
    bool stable_detected = false;

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("%s: %.5f\r\n", result.classification[ix].label, result.classification[ix].value);
        if (strcmp(result.classification[ix].label, "down") == 0 && result.classification[ix].value > 0.8f)
            down_detected = true;
        if (strcmp(result.classification[ix].label, "stable") == 0 && result.classification[ix].value > 0.8f)
            stable_detected = true;
    }

    if (down_detected && !buzzer_on) {
        digitalWrite(BUZZER_PIN, HIGH);
        buzzer_on = true;
        ei_printf("Down detected - Buzzer ON\r\n");

        if (!call_in_progress) {
            make_gsm_call(PHONE_NUMBER);
            call_in_progress = true;
            ei_printf("GSM Call initiated.\r\n");
        }
    }

    if (digitalRead(BUTTON_PIN) == LOW && buzzer_on) {
        digitalWrite(BUZZER_PIN, LOW);
        buzzer_on = false;
        ei_printf("Button pressed - Buzzer OFF\r\n");
    }

    digitalWrite(LED_BUILTIN, down_detected ? HIGH : (stable_detected ? LOW : digitalRead(LED_BUILTIN)));

    if (stable_detected) {
        call_in_progress = false; // Reset for next detection
    }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly score: %.3f\r\n", result.anomaly);
#endif
}

float ei_get_sign(float number) {
    return (number >= 0.0) ? 1.0 : -1.0;
}

bool init_IMU() {
    static bool s = false;
    if (!s) s = IMU.begin();
    return s;
}

bool init_HTS() {
    static bool s = false;
    if (!s) s = HS300x.begin();
    return s;
}

bool init_BARO() {
    static bool s = false;
    if (!s) s = BARO.begin();
    return s;
}

bool init_APDS() {
    static bool s = false;
    if (!s) s = APDS.begin();
    return s;
}

uint8_t poll_acc() {
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(data[0], data[1], data[2]);
        for (int i = 0; i < 3; i++) {
            if (fabs(data[i]) > MAX_ACCEPTED_RANGE)
                data[i] = ei_get_sign(data[i]) * MAX_ACCEPTED_RANGE;
        }
        data[0] *= CONVERT_G_TO_MS2;
        data[1] *= CONVERT_G_TO_MS2;
        data[2] *= CONVERT_G_TO_MS2;
    }
    return 0;
}

uint8_t poll_gyr() {
    if (IMU.gyroscopeAvailable()) IMU.readGyroscope(data[3], data[4], data[5]);
    return 0;
}

uint8_t poll_mag() {
    if (IMU.magneticFieldAvailable()) IMU.readMagneticField(data[6], data[7], data[8]);
    return 0;
}

uint8_t poll_HTS() {
    data[9] = HS300x.readTemperature();
    data[10] = HS300x.readHumidity();
    return 0;
}

uint8_t poll_BARO() {
    data[11] = BARO.readPressure();
    return 0;
}

uint8_t poll_APDS_color() {
    int temp_data[4];
    if (APDS.colorAvailable()) {
        APDS.readColor(temp_data[0], temp_data[1], temp_data[2], temp_data[3]);
        data[12] = temp_data[0];
        data[13] = temp_data[1];
        data[14] = temp_data[2];
        data[15] = temp_data[3];
    }
    return 0;
}

uint8_t poll_APDS_proximity() {
    if (APDS.proximityAvailable())
        data[16] = (float)APDS.readProximity();
    return 0;
}

uint8_t poll_APDS_gesture() {
    if (APDS.gestureAvailable())
        data[17] = (float)APDS.readGesture();
    return 0;
}

int8_t ei_find_axis(char *axis_name) {
    for (int ix = 0; ix < N_SENSORS; ix++) {
        if (strstr(axis_name, sensors[ix].name))
            return ix;
    }
    return -1;
}

bool ei_connect_fusion_list(const char *input_list) {
    char *input_string = (char *)ei_malloc(strlen(input_list) + 1);
    if (!input_string) return false;
    memset(input_string, 0, strlen(input_list) + 1);
    strncpy(input_string, input_list, strlen(input_list));

    memset(fusion_sensors, 0, N_SENSORS);
    fusion_ix = 0;

    char *buff = strtok(input_string, "+");
    while (buff != NULL) {
        int8_t found_axis = ei_find_axis(buff);
        if (found_axis >= 0 && fusion_ix < N_SENSORS) {
            fusion_sensors[fusion_ix++] = found_axis;
            sensors[found_axis].status = NOT_INIT;
        }
        buff = strtok(NULL, "+ ");
    }

    ei_free(input_string);
    return fusion_ix > 0;
}
void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:

}
