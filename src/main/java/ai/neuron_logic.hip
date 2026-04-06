#include "ai_Neuron.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <jni.h>
#include <thread>
#include <numeric>
#include <map>

extern "C"
{
    void sum_worker(const std::vector<double>& w, int start, int end, double* partial_sum) {
        double local_sum = 0.0;
        for (int i = start; i < end; ++i) {
            local_sum += w[i];
        }
        *partial_sum = local_sum;
    }

    double get_multithreaded_sum(const std::vector<double>& w) {
        const int num_threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
        int len = w.size();
        if (len < 1000) {
            double s = 0.0;
            for (double val : w) {
                s += val;
            }
            return s;
        }

        std::vector<std::thread> threads;
        std::vector<double> partial_results(num_threads, 0.0);
        int chunk = len / num_threads;

        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk;
            int end = (i == num_threads - 1) ? len : (i + 1) * chunk;
            threads.emplace_back(sum_worker, std::ref(w), start, end, &partial_results[i]);
        }

        double total_sum = 0.0;
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
            total_sum += partial_results[i];
        }
        return total_sum;
    }

    JNIEXPORT jdouble JNICALL Java_ai_Neuron_activate(JNIEnv *env, jclass cls, jdouble x) {
        return std::tanh(x);
    }

    JNIEXPORT jdouble JNICALL Java_ai_Neuron_derivative(JNIEnv *env, jclass cls, jdouble x) {
        return 1.0 - (x * x);
    }

    JNIEXPORT jdouble JNICALL Java_ai_Neuron_predict(JNIEnv *env, jobject obj) {
        jclass cls = env->GetObjectClass(obj);
        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        double b = env->GetDoubleField(obj, bF);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL) {
            return 0.0;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        double s = get_multithreaded_sum(w);
        s += b;
        return std::tanh(s);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_train(JNIEnv *env, jobject obj, jdouble target) {
        jclass cls = env->GetObjectClass(obj);
    
        jfieldID peersF = env->GetFieldID(cls, "peers", "[Lai/Neuron;");
        jobjectArray peersA = (jobjectArray)env->GetObjectField(obj, peersF);
    
        double collectiveError = 0.0;
        if (peersA != NULL) {
            jsize peerCount = env->GetArrayLength(peersA);
            double totalPeerOut = 0.0;

            for (int i = 0; i < peerCount; i++) {
                jobject peer = env->GetObjectArrayElement(peersA, i);
                jclass peerCls = env->GetObjectClass(peer);
                jmethodID predictID = env->GetMethodID(peerCls, "predict", "()D");
                totalPeerOut += env->CallDoubleMethod(peer, predictID);
            }
        
            double avgPeerOut = totalPeerOut / (double)peerCount;
            collectiveError = target - avgPeerOut;
        }

        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        double b = env->GetDoubleField(obj, bF);

        jfieldID lrF = env->GetFieldID(cls, "learningRate", "D");
        double lr = env->GetDoubleField(obj, lrF);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL) {
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        double myOut = Java_ai_Neuron_predict(env, obj);
        double myError = target - myOut;

        double derivative = 1.0 - (myOut * myOut);

        double errorSignal = (peersA != NULL) ? (myError - collectiveError) : myError;

        double delta = errorSignal * derivative;

        for (jsize i = 0; i < len; ++i) {
            w[i] += lr * delta;
        }
        b += lr * delta;

        std::map<double, double> loyalties;
        const double pi = 3.141592653589793;

        for (jsize i = 0; i < len; i++) {
            loyalties.insert(std::make_pair(w[i], w[i] * pi));
        }

        double bad_l = 0; 
        double big_l = 0;
        bool first = true;

        for (jsize i = 0; i < len; i++) {
            double current_loyalty = loyalties[w[i]];
            if (first) {
                bad_l = big_l = current_loyalty;
                first = false;
            }
            if (current_loyalty < bad_l) {
                bad_l = current_loyalty;
            }
            if (current_loyalty > big_l) {
                big_l = current_loyalty;
            }
        }

        for (jsize i = 0; i < len; i++) {
            w[i] = w[i] * (big_l != 0 ? (1.0 / big_l) : 1.0) * (bad_l != 0 ? (1.0 / bad_l) : 1.0);
        }

        env->SetDoubleArrayRegion(wA, 0, len, w.data());
        env->SetDoubleField(obj, bF, b);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_motivate(JNIEnv *env, jobject obj, jdouble reward, jdouble gamma, jdouble state) {
        jclass cls = env->GetObjectClass(obj);
        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        double b = env->GetDoubleField(obj, bF);

        jfieldID lrF = env->GetFieldID(cls, "learningRate", "D");
        double lr = env->GetDoubleField(obj, lrF);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL) {
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        double out = Java_ai_Neuron_predict(env, obj);
        double trg = reward + gamma * state;
        double err = trg - out;

        double deriv = 1.0 - (out * out);

        for (jsize i = 0; i < len; ++i) {
            w[i] += lr * err * deriv;
        }
        b += lr * err * deriv;

        env->SetDoubleArrayRegion(wA, 0, len, w.data());
        env->SetDoubleField(obj, bF, b);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_exportWeights(JNIEnv *env, jobject obj, jstring filename) {
        const char *nm = env->GetStringUTFChars(filename, 0);
        jclass cls = env->GetObjectClass(obj);

        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        double b = env->GetDoubleField(obj, bF);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL) {
            env->ReleaseStringUTFChars(filename, nm);
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        std::ofstream f(nm);
        for (jsize i = 0; i < len; ++i) {
            f << w[i] << " ";
        }
        f << b << std::endl;
        f.close();

        env->ReleaseStringUTFChars(filename, nm);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_importWeights(JNIEnv *env, jobject obj, jstring filename) {
        const char *nm = env->GetStringUTFChars(filename, 0);
        jclass cls = env->GetObjectClass(obj);

        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL) {
            env->ReleaseStringUTFChars(filename, nm);
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        
        std::ifstream f(nm);
        if (f.is_open()) {
            for (jsize i = 0; i < len; ++i) {
                f >> w[i];
            }
            double b;
            f >> b;
            f.close();

            env->SetDoubleArrayRegion(wA, 0, len, w.data());
            env->SetDoubleField(obj, bF, b);
        }

        env->ReleaseStringUTFChars(filename, nm);
    }
}