/*
  This code is based on a example from Fraunhofer-IMS/AIfES_for_Arduino, updated for readability
  /examples/0_Universal/0_XOR_F32/1_XOR_F32_training/1_XOR_F32_training.ino

  it can be found here: https://github.com/Fraunhofer-IMS/AIfES_for_Arduino/blob/main/examples/0_Universal/0_XOR_F32/1_XOR_F32_training/1_XOR_F32_training.ino

  Copyright (C) 2020-2023  Fraunhofer Institute for Microelectronic Circuits and Systems.
  Copyright (C) 2024 Chris Dirks
  All rights reserved.

  AIfES is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include <aifes.h>  // include the AIfES libary

/**
 * AIfES XOR Gate training demo
 * 
 * This sketch demonstrates training a neural network from scratch using AIfES and training data.
 * The network structure is 2-3(Sigmoid)-1(Sigmoid) with Sigmoid as the activation function.
 * 
 * Tested on: Arduino UNO, Nano, Nano 33 BLE, SAMD21, ESP32
 */

// Function declarations
void customPrintModelStructure(aimodel_t *model);
void customPrintMemoryRequired(uint32_t parameter_memory_size, uint32_t memory_size, byte *memory_ptr);
void customPrintTrainingSetup(uint16_t epochs, uint32_t batch_size, aiopti_t *optimizer, aimodel_t &model);
void customPrintEpochOnInterval(uint32_t i, uint16_t print_interval, aimodel_t &model, aitensor_t &input_tensor, aitensor_t &target_tensor, float &loss, float &prevLoss, u_int32_t startTime, uint16_t epochs);
void customPrintModelEvaluation(float target_data[4], float model_output_data[4], float input_data[4][2]);


void setup() {
    Serial.begin(115200);  //115200 baud rate (If necessary, change in the serial monitor)
    while (!Serial);       //Wait for someone to open the serial monitor

    // ---------------------------------- #1.1 Setup the Training Data ---------------------------------------
    // XOR function Inputs and Expected Output (Binary translated to floats)
    // XOR inputs -> input Tensor
    uint16_t input_shape[] = { 4, 2 };
    float input_data[4][2] = {
        { 0.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 0.0f },
        { 1.0f, 1.0f }
    };
    // XOR correct (target) outputs -> target Tensor
    uint16_t target_shape[] = { 4, 1 };
    float target_data[4 * 1] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);     // Creation of the input AIfES tensor with two dimensions and data type F32 (float32)
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);  // Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored


    // --------------------------- #1.2 Setup the structure of the model ----------------------------
    aimodel_t model;  // AIfES model
    ailayer_t *x;     // Layer object from AIfES to connect the layers

    uint16_t input_layer_shape[] = { 1, 2 };                                                                          // Definition of the input layer shape (Must fit to the input tensor)
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(/*input dimension=*/2, /*input shape=*/input_layer_shape);  // Creation of the AIfES input layer
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(/*neurons=*/3);                                           // Creation of the AIfES hidden dense layer with 3 neurons
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_A();                                                  // Hidden activation function
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(/*neurons=*/1);                                           // Creation of the AIfES output dense layer with 1 neuron
    ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_A();                                                  // Output activation function

    ailoss_mse_t mse_loss;  //Loss: mean squared error

    // Connect the layers to an AIfES model
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    model.output_layer = x;

    model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);  // Add the loss algorithm to the AIfES model
    aialgo_compile_model(&model);                                        // Compile the AIfES model

    customPrintModelStructure(&model);


    // ------------------------------- #1.3 Setup the Neuron Weights and Biases ------------------------------
    //Allocate memory for the Neuron Weights and Biases
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    byte *parameter_memory = (byte *)malloc(parameter_memory_size);
    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);  // Distribute the memory for the trainable parameters of the model

    //Set the values for the Neuron Weights and Biases
    srand(analogRead(A5));  //IMPORTANT AIfES requires random weights for training, Here the random seed is generated by the noise of an analog pin
    aimath_f32_default_init_glorot_uniform(&dense_layer_1.weights);
    aimath_f32_default_init_zeros(&dense_layer_1.bias);
    aimath_f32_default_init_glorot_uniform(&dense_layer_2.weights);
    aimath_f32_default_init_zeros(&dense_layer_2.bias);


    // -------------------------------- #1.4 Setup the training and optimizer (default: Adam algorithm, Alternatives: SGD, SGD with Momentum) ---------------------
    aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(/*learning rate=*/0.1f, /*beta_1=*/0.9f, /*beta_2=*/0.999f, /*eps=*/1e-7);
    aiopti_t *optimizer = aiopti_adam_f32_default(&adam_opti);  // Initialize the optimizer

    // Allocate and schedule the working memory
    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    byte *memory_ptr = (byte *)malloc(memory_size);
    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);  // Schedule the memory over the model
    aialgo_init_model_for_training(&model, optimizer);                            // IMPORTANT: Initialize the AIfES model before training

    customPrintMemoryRequired(parameter_memory_size, memory_size, memory_ptr);


    // ------------------------------------- #2 Train the Model ------------------------------------
    uint32_t batch_size = 4;       // Number of Data points (from the training data) to train on in each epoch
    uint16_t epochs = 100;         // Number of training batches
    uint16_t print_interval = 25;  //after how many new epochs to print to serial
    customPrintTrainingSetup(epochs, batch_size, optimizer, model);

    float loss, prevLoss;
    aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &prevLoss);  //calculate loss for untrained model
    u_int32_t startTime = millis();

    for (int i = 1; i <= epochs; i++) {
        aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);  // One epoch of training. Iterates through the whole data once
        customPrintEpochOnInterval(i, print_interval, model, input_tensor, target_tensor, loss, prevLoss, startTime, epochs);
    }


    // ----------------------------------------- #3 Evaluate the trained model --------------------------
    uint16_t output_shape[] = { 4, 1 };
    float model_output_data[4 * 1];  // Empty tensor for the model test output data (Must have the same configuration as for the target tensor)
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, model_output_data);

    aialgo_inference_model(&model, &input_tensor, &output_tensor);  // Take the XOR inputs and calculate the model outputs
    customPrintModelEvaluation(target_data, model_output_data, input_data);


    free(parameter_memory);
    free(memory_ptr);
}

void loop() {
    delay(1);
}

void customPrintModelEvaluation(float target_data[4], float model_output_data[4], float input_data[4][2]) {
    float averageModelCorrectness = 0;
    Serial.printf("\n------------ Model Evaluation ----------\n");
    for (int i = 0; i < 4; i++) {
        float questionCorrectnessPercent = 100.0 - (abs(target_data[i] - model_output_data[i]) * 100.0);
        averageModelCorrectness += questionCorrectnessPercent / 4;
        Serial.printf("XOR(%0.0f,%0.0f) = %0.0f ≈ %0.2f (%0.1F%% Correct)\n", input_data[i][0], input_data[i][1], target_data[i], model_output_data[i], questionCorrectnessPercent);
    }

    Serial.printf("\nYour model in on Average %0.1F%% Correct :)\n", averageModelCorrectness);
}

void customPrintEpochOnInterval(uint32_t i, uint16_t print_interval, aimodel_t &model, aitensor_t &input_tensor, aitensor_t &target_tensor, float &loss, float &prevLoss, u_int32_t startTime, uint16_t epochs) {
    if (i % print_interval == 0) {  // Calculate and print loss every print_interval epochs
        aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
        if (prevLoss < 0.0) prevLoss = loss;
        float lossImprovementPerEpoch = (((prevLoss - loss) / prevLoss) / print_interval) * 100.0;
        float speed = float(millis() - startTime) / (float)i;
        Serial.printf("%05ums \tEpoch: %03u/%u \tLoss: %0.3f  \tImprovement ≈ %+0.2f%%/epoch \tSpeed: %0.1fms/epoch\n", millis() - startTime, i, epochs, loss, lossImprovementPerEpoch, speed);
        prevLoss = loss;
    }
}

void customPrintTrainingSetup(uint16_t epochs, uint32_t batch_size, aiopti_t *optimizer, aimodel_t &model) {
    Serial.printf("\n------------ Model Training ----------\n");
    Serial.printf("Epochs: %u \tBatch size: %u \tOptimizer: ", epochs, batch_size);
    aialgo_print_optimizer_specs(optimizer);
    Serial.printf("\nLoss (scoring algorithm): ");
    aialgo_print_loss_specs(model.loss);
    Serial.print("\n\n");
}

void customPrintModelStructure(aimodel_t *model) {
    Serial.println("\n-------------- Model structure ---------------");
    ailayer_t *layer_ptr = model->input_layer;
    for (int i = 0; i < model->layer_count; i++) {
        if (layer_ptr->layer_type->print_specs != 0) {
            Serial.printf("%02u: %s Layer (%s), Specs{", i + 1, layer_ptr->layer_type->name, layer_ptr->result.dtype->name);
            layer_ptr->layer_type->print_specs(layer_ptr);
            Serial.println("}");
        } else {
            Serial.printf("%02u: No specs found for this layer.\n", i + 1);
        }
        layer_ptr = layer_ptr->output_layer;
    }
    return;
}

void customPrintMemoryRequired(uint32_t parameter_memory_size, uint32_t memory_size, byte *memory_ptr) {
    Serial.printf("\n-------------- Memory Required ---------------\n");
    Serial.printf("Required memory for parameter (Weights, Biases): %u bytes\n", parameter_memory_size);
    Serial.printf("Training (Intermediate results, gradients, optimization memory): %u bytes\n", memory_size);
    if (memory_ptr == 0) {
        Serial.print("\nERROR: Not enough memory (RAM) available for training!\n Try to use another optimizer (e.g. SGD) or make your net smaller.");
        while (true);
    }
}
