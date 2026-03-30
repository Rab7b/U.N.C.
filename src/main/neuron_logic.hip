#include <jni.h>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "ai_Neuron.h"

static double alpha = 0.01;

__device__ double gpu_activate(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

__device__ double gpu_derivative(double x) {
    return x * (1.0 - x);
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
    jsize numRows = env->GetArrayLength(weightsArray);
    for (int i = 0; i < numRows; i++) {
        jdoubleArray row = (jdoubleArray)env->GetObjectArrayElement(weightsArray, i);
        jsize numCols = env->GetArrayLength(row);
        jdouble* elements = env->GetDoubleArrayElements(row, nullptr);
        for (int j = 0; j < numCols; j++) {
            file << std::fixed << std::setprecision(6) << (double)elements[j];
            if (j < numCols - 1) file << " ";
        }
        file << "\n";
        env->ReleaseDoubleArrayElements(row, elements, JNI_ABORT);
        env->DeleteLocalRef(row);
    }
    file.close();
    env->ReleaseStringUTFChars(path, nativePath);
}

JNIEXPORT jdouble JNICALL Java_ai_Neuron_activate(JNIEnv* env, jclass cls, jdouble x) {
    double h_res;
    double *d_x, *d_res;
    hipMalloc(&d_x, sizeof(double));
    hipMalloc(&d_res, sizeof(double));
    hipMemcpy(d_x, &x, sizeof(double), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(kernel_activate, dim3(1), dim3(1), 0, 0, d_x, d_res);
    hipMemcpy(&h_res, d_res, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_x); hipFree(d_res);
    return (jdouble)h_res;
}

JNIEXPORT jdouble JNICALL Java_ai_Neuron_derivative(JNIEnv* env, jclass cls, jdouble x) {
    double h_res;
    double *d_x, *d_res;
    hipMalloc(&d_x, sizeof(double));
    hipMalloc(&d_res, sizeof(double));
    hipMemcpy(d_x, &x, sizeof(double), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(kernel_derivative, dim3(1), dim3(1), 0, 0, d_x, d_res);
    hipMemcpy(&h_res, d_res, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_x); hipFree(d_res);
    return (jdouble)h_res;
}

JNIEXPORT jdouble JNICALL Java_ai_Neuron_predict(JNIEnv* env, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    jdoubleArray jInputs = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "inputs", "[D"));
    jobjectArray jWeights = (jobjectArray)env->GetObjectField(obj, env->GetFieldID(cls, "weights", "[[D"));
    jdoubleArray jBias = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "bias", "[D"));
    jdoubleArray jOutField = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "out", "[D"));
    
    jsize inputLen = env->GetArrayLength(jInputs);
    jdouble* inputsPtr = env->GetDoubleArrayElements(jInputs, nullptr);
    jdouble* biasPtr = env->GetDoubleArrayElements(jBias, nullptr);
    
    std::vector<double> out(7, 0.0);
    double sum1 = biasPtr[0];
    for (int i = 0; i < inputLen; i++) {
        jdoubleArray r = (jdoubleArray)env->GetObjectArrayElement(jWeights, i);
        jdouble* re = env->GetDoubleArrayElements(r, nullptr);
        sum1 += inputsPtr[i] * re[0];
        env->ReleaseDoubleArrayElements(r, re, JNI_ABORT);
        env->DeleteLocalRef(r);
    }

    double *d_s, *d_a;
    hipMalloc(&d_s, sizeof(double));
    hipMalloc(&d_a, sizeof(double));
    hipMemcpy(d_s, &sum1, sizeof(double), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(kernel_activate, dim3(1), dim3(1), 0, 0, d_s, d_a);
    hipMemcpy(&out[0], d_a, sizeof(double), hipMemcpyDeviceToHost);

    jdoubleArray row0 = (jdoubleArray)env->GetObjectArrayElement(jWeights, 0);
    jdouble* r0e = env->GetDoubleArrayElements(row0, nullptr);

    for (int j = 1; j < 7; j++) {
        double val = biasPtr[j] + out[j-1] * r0e[j];
        hipMemcpy(d_s, &val, sizeof(double), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(kernel_activate, dim3(1), dim3(1), 0, 0, d_s, d_a);
        hipMemcpy(&out[j], d_a, sizeof(double), hipMemcpyDeviceToHost);
    }
    
    hipFree(d_s); hipFree(d_a);
    env->ReleaseDoubleArrayElements(row0, r0e, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jInputs, inputsPtr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jBias, biasPtr, JNI_ABORT);
    
    for(int i=0; i<7; i++) if(std::isnan(out[i])) out[i] = 0.5;
    env->SetDoubleArrayRegion(jOutField, 0, 7, out.data());
    return (jdouble)out[6];
}

JNIEXPORT void JNICALL Java_ai_Neuron_train(JNIEnv *env, jobject obj, jdouble target) {
    jclass cls = env->GetObjectClass(obj);
    jmethodID mid = env->GetMethodID(cls, "predict", "()D");
    env->CallDoubleMethod(obj, mid);

    jdoubleArray oA = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "out", "[D"));
    jdoubleArray iA = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "inputs", "[D"));
    jdoubleArray bA = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "bias", "[D"));
    jobjectArray wA = (jobjectArray)env->GetObjectField(obj, env->GetFieldID(cls, "weights", "[[D"));
    double lr = env->GetDoubleField(obj, env->GetFieldID(cls, "learningRate", "D"));

    double* co = env->GetDoubleArrayElements(oA, NULL);
    double* ci = env->GetDoubleArrayElements(iA, NULL);
    double* cb = env->GetDoubleArrayElements(bA, NULL);

    double g[7];
    g[6] = (target - co[6]) * co[6] * (1.0 - co[6]);
    jdoubleArray r0 = (jdoubleArray)env->GetObjectArrayElement(wA, 0);
    double* r0e = env->GetDoubleArrayElements(r0, NULL);
    for(int j=5; j>=0; j--) g[j] = (g[j+1] * r0e[j+1]) * co[j] * (1.0 - co[j]);
    env->ReleaseDoubleArrayElements(r0, r0e, JNI_ABORT);

    int rows = env->GetArrayLength(wA);
    for(int i=0; i<rows; i++){
        jdoubleArray row = (jdoubleArray)env->GetObjectArrayElement(wA, i);
        double* re = env->GetDoubleArrayElements(row, NULL);
        re[0] += lr * g[0] * ci[i];
        for(int j=1; j<7; j++) re[j] += lr * g[j] * co[j-1];
        env->ReleaseDoubleArrayElements(row, re, 0);
        env->DeleteLocalRef(row);
    }
    for(int i=0; i<7; i++) cb[i] += lr * g[i];
    env->ReleaseDoubleArrayElements(oA, co, JNI_ABORT);
    env->ReleaseDoubleArrayElements(iA, ci, JNI_ABORT);
    env->ReleaseDoubleArrayElements(bA, cb, 0);
}

JNIEXPORT void JNICALL Java_ai_Neuron_motivate(JNIEnv *env, jobject obj, jdouble reward, jdoubleArray nextState, jdouble gamma) {
    jclass cls = env->GetObjectClass(obj);
    jfieldID fid = env->GetFieldID(cls, "inputs", "[D");
    jdoubleArray old = (jdoubleArray)env->GetObjectField(obj, fid);
    env->SetObjectField(obj, fid, nextState);
    double nQ = env->CallDoubleMethod(obj, env->GetMethodID(cls, "predict", "()D"));
    env->SetObjectField(obj, fid, old);
    env->CallVoidMethod(obj, env->GetMethodID(cls, "train", "(D)V"), (jdouble)(reward + gamma * nQ));
}