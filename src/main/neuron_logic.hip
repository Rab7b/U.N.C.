#include <jni.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <execution>
#include <fstream>
#include <string>
#include <iostream>
#include <cmath>
#include <thread>
#include <sstream> 
#include <iomanip>
#include <hip/hip_runtime.h>
#include "ai_Neuron.h"

__device__ float relu_alpha = 0.01f;

__device__ double gpu_activate(double x) {
    return (x > 0.0) ? x : alpha * (exp(x) - 1.0);
}

__device__ double gpu_derivative_elu(double x, double alpha) {
    if (x > 0.0) {
        return 1.0;
    } else {
        return alpha * exp(x);
    }
}

__global__ void kernel_activate(double* x, double* res) {
    *res = gpu_activate(*x);
}

__global__ void kernel_derivative(double* x, double* res) {
    *res = gpu_derivative(*x);
}

JNIEXPORT void JNICALL Java_ai_Neuron_importWeights(JNIEnv* env, jobject obj, jstring path) {
    const char* nativePath = env->GetStringUTFChars(path, nullptr);
    if (nativePath == nullptr) return;
    std::ifstream file(nativePath);
    if (!file.is_open()) {
        env->ReleaseStringUTFChars(path, nativePath);
        return;
    }
    std::string line;
    while (std::getline(file, line)) {}
    file.close();
    env->ReleaseStringUTFChars(path, nativePath);
}

JNIEXPORT void JNICALL Java_ai_Neuron_exportWeights(JNIEnv* env, jobject obj, jstring path, jobjectArray weightsArray) {
    const char* nativePath = env->GetStringUTFChars(path, nullptr);
    if (nativePath == nullptr) return;
    std::ofstream file(nativePath);
    if (!file.is_open()) {
        env->ReleaseStringUTFChars(path, nativePath);
        return;
    }
    jsize numRows = env->GetArrayLength(weightsArray);
    for (int i = 0; i < numRows; i++) {
        jdoubleArray row = (jdoubleArray)env->GetObjectArrayElement(weightsArray, i);
        jsize numCols = env->GetArrayLength(row);
        jdouble* elements = env->GetDoubleArrayElements(row, nullptr);
        for (int j = 0; j < numCols; j++) {
            file << std::fixed << std::setprecision(6) << elements[j];
            if (j < numCols - 1) file << " ";
        }
        file << "\n";
        env->ReleaseDoubleArrayElements(row, elements, JNI_ABORT);
        env->DeleteLocalRef(row); 
    }
    file.close();
    env->ReleaseStringUTFChars(path, nativePath);
}

JNIEXPORT jdouble JNICALL Java_ai_Neuron_activate(JNIEnv* env, jobject obj, jdouble x) {
    double h_x = (double)x;
    double h_res;
    double *d_x, *d_res;
    hipMalloc(&d_x, sizeof(double));
    hipMalloc(&d_res, sizeof(double));
    hipMemcpy(d_x, &h_x, sizeof(double), hipMemcpyHostToDevice);
    kernel_activate<<<dim3(1), dim3(1), 0, 0>>>(d_x, d_res);
    hipMemcpy(&h_res, d_res, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_x);
    hipFree(d_res);
    return (jdouble)h_res;
}

JNIEXPORT jdouble JNICALL Java_ai_Neuron_derivative(JNIEnv* env, jobject obj, jdouble x) {
    double h_x = (double)x;
    double h_res;
    double *d_x, *d_res;
    hipMalloc(&d_x, sizeof(double));
    hipMalloc(&d_res, sizeof(double));
    hipMemcpy(d_x, &h_x, sizeof(double), hipMemcpyHostToDevice);
    kernel_derivative<<<dim3(1), dim3(1), 0, 0>>>(d_x, d_res);
    hipMemcpy(&h_res, d_res, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_x);
    hipFree(d_res);
    return (jdouble)h_res;
}

JNIEXPORT jdouble JNICALL Java_ai_Neuron_predict(JNIEnv* env, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    jdoubleArray jInputs = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "inputs", "[D"));
    jobjectArray jWeights = (jobjectArray)env->GetObjectField(obj, env->GetFieldID(cls, "weights", "[[D"));
    jdoubleArray jBias = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "bias", "[D"));
    jsize inputLen = env->GetArrayLength(jInputs);
    jdouble* inputsPtr = env->GetDoubleArrayElements(jInputs, nullptr);
    jdouble* biasPtr = env->GetDoubleArrayElements(jBias, nullptr);
    std::vector<double> inputs(inputsPtr, inputsPtr + inputLen);
    std::vector<double> bias(biasPtr, biasPtr + 7);
    std::vector<double> out(7, 0.0);
    jdoubleArray row0 = (jdoubleArray)env->GetObjectArrayElement(jWeights, 0);
    jdouble* row0E = env->GetDoubleArrayElements(row0, nullptr);
    std::vector<double> wCol0(inputLen);
    for(int i=0; i<inputLen; i++) {
        jdoubleArray r = (jdoubleArray)env->GetObjectArrayElement(jWeights, i);
        jdouble* re = env->GetDoubleArrayElements(r, nullptr);
        wCol0[i] = re[0];
        env->ReleaseDoubleArrayElements(r, re, JNI_ABORT);
        env->DeleteLocalRef(r);
    }
    double sum1 = bias[0];
    for (int i = 0; i < inputLen; i++) {
        sum1 += inputs[i] * wCol0[i];
    }
    
    double h_sum1 = sum1;
    double h_act1;
    double *d_s1, *d_a1;
    hipMalloc(&d_s1, sizeof(double));
    hipMalloc(&d_a1, sizeof(double));
    hipMemcpy(d_s1, &h_sum1, sizeof(double), hipMemcpyHostToDevice);
    kernel_activate<<<dim3(1), dim3(1), 0, 0>>>(d_s1, d_a1);
    hipMemcpy(&h_act1, d_a1, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_s1);
    hipFree(d_a1);
    out[0] = h_act1;

    for (int j = 1; j < 7; j++) {
        double val = bias[j] + out[j-1] * row0E[j];
        double h_act_loop;
        double *d_val, *d_res_loop;
        hipMalloc(&d_val, sizeof(double));
        hipMalloc(&d_res_loop, sizeof(double));
        hipMemcpy(d_val, &val, sizeof(double), hipMemcpyHostToDevice);
        kernel_activate<<<dim3(1), dim3(1), 0, 0>>>(d_val, d_res_loop);
        hipMemcpy(&h_act_loop, d_res_loop, sizeof(double), hipMemcpyDeviceToHost);
        hipFree(d_val);
        hipFree(d_res_loop);
        out[j] = h_act_loop;
    }
    env->ReleaseDoubleArrayElements(row0, row0E, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jInputs, inputsPtr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jBias, biasPtr, JNI_ABORT);
    return (jdouble)out[6];
}

JNIEXPORT void JNICALL Java_ai_Neuron_train(JNIEnv *env, jobject obj, jdouble target) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID wID = env->GetFieldID(cls, "weights", "[[D");
    jfieldID iID = env->GetFieldID(cls, "inputs", "[D");
    jfieldID oID = env->GetFieldID(cls, "out", "[D");
    jfieldID bID = env->GetFieldID(cls, "bias", "[D");
    jfieldID lrID = env->GetFieldID(cls, "learningRate", "D");
    jmethodID predictID = env->GetMethodID(cls, "predict", "()D");
    env->CallDoubleMethod(obj, predictID);
    jobjectArray weights = (jobjectArray)env->GetObjectField(obj, wID);
    jdoubleArray outArr = (jdoubleArray)env->GetObjectField(obj, oID);
    jdoubleArray inputsArr = (jdoubleArray)env->GetObjectField(obj, iID);
    jdoubleArray biasArr = (jdoubleArray)env->GetObjectField(obj, bID);
    double learningRate = env->GetDoubleField(obj, lrID);
    double* cOut = env->GetDoubleArrayElements(outArr, NULL);
    double* cInputs = env->GetDoubleArrayElements(inputsArr, NULL);
    double* cBias = env->GetDoubleArrayElements(biasArr, NULL);
    double gradients[7] = {0.0};
    gradients[6] = (target - cOut[6]) * (1.5 * cOut[6] * cOut[6]);
    int totalRows = env->GetArrayLength(weights);
    jdoubleArray row0 = (jdoubleArray)env->GetObjectArrayElement(weights, 0);
    double* row0Elements = env->GetDoubleArrayElements(row0, NULL);
    for (int j = 5; j >= 0; j--) {
        gradients[j] = (gradients[j + 1] * row0Elements[j + 1]) * (1.5 * cOut[j] * cOut[j]);
    }
    env->ReleaseDoubleArrayElements(row0, row0Elements, JNI_ABORT);
    env->DeleteLocalRef(row0);
    for (int i = 0; i < totalRows; i++) {
        jdoubleArray rowArr = (jdoubleArray)env->GetObjectArrayElement(weights, i);
        double* row = env->GetDoubleArrayElements(rowArr, NULL);
        row[0] += learningRate * gradients[0] * cInputs[i];
        for (int j = 1; j < 7; j++) {
            row[j] += learningRate * gradients[j] * cOut[j - 1];
        }
        env->ReleaseDoubleArrayElements(rowArr, row, 0);
        env->DeleteLocalRef(rowArr);
    }
    for (int i = 0; i < 7; i++) {
        cBias[i] += learningRate * gradients[i];
    }
    env->ReleaseDoubleArrayElements(outArr, cOut, JNI_ABORT);
    env->ReleaseDoubleArrayElements(inputsArr, cInputs, JNI_ABORT);
    env->ReleaseDoubleArrayElements(biasArr, cBias, 0);
}

JNIEXPORT void JNICALL Java_ai_Neuron_motivate(JNIEnv *env, jobject obj, jdouble reward, jdoubleArray nextState, jdouble gamma) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID inputsID = env->GetFieldID(cls, "inputs", "[D");
    jmethodID predictID = env->GetMethodID(cls, "predict", "()D");
    jmethodID trainID = env->GetMethodID(cls, "train", "(D)V");
    jdoubleArray oldStateArr = (jdoubleArray)env->GetObjectField(obj, inputsID);
    env->SetObjectField(obj, inputsID, nextState);
    double nextMaxQ = env->CallDoubleMethod(obj, predictID);
    double targetValue = reward + gamma * nextMaxQ;
    env->SetObjectField(obj, inputsID, oldStateArr);
    env->CallVoidMethod(obj, trainID, (jdouble)targetValue);
}