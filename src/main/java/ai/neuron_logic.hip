#include "ai_Neuron.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <jni.h>
#include <thread>
#include <numeric>
#include <ctime>
#include <random>
#include <map>

const double PI = 3.141592653589793;
const double SPARTAN_LIMIT = 0.15;

extern "C"
{
    void sum_worker(const std::vector<double> &w, int start, int end, double *partial_sum)
    {
        double local_sum = 0.0;
        for (int i = start; i < end; ++i)
        {
            local_sum += w[i];
        }
        *partial_sum = local_sum;
    }

    double get_multithreaded_sum(const std::vector<double> &w)
    {
        const int num_threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;
        int len = w.size();
        if (len < 1000)
        {
            double s = 0.0;
            for (double val : w)
            {
                s += val;
            }
            return s;
        }

        std::vector<std::thread> threads;
        std::vector<double> partial_results(num_threads, 0.0);
        int chunk = len / num_threads;

        for (int i = 0; i < num_threads; ++i)
        {
            int start = i * chunk;
            int end = (i == num_threads - 1) ? len : (i + 1) * chunk;
            threads.emplace_back(sum_worker, std::ref(w), start, end, &partial_results[i]);
        }

        double total_sum = 0.0;
        for (int i = 0; i < num_threads; ++i)
        {
            threads[i].join();
            total_sum += partial_results[i];
        }
        return total_sum;
    }

    JNIEXPORT jdouble JNICALL Java_ai_Neuron_activate(JNIEnv *env, jclass cls, jdouble x)
    {
        return std::tanh(x);
    }

    JNIEXPORT jdouble JNICALL Java_ai_Neuron_predict(JNIEnv *env, jobject obj)
    {
        jclass cls = env->GetObjectClass(obj);

        double b = env->GetDoubleField(obj, env->GetFieldID(cls, "bias", "D"));

        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "weights", "[D"));
        if (wA == NULL)
            return 0.0;
        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        jdoubleArray iA = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "inputs", "[D"));
        if (iA == NULL)
            return std::tanh(b);
        std::vector<double> ins(len);
        env->GetDoubleArrayRegion(iA, 0, len, ins.data());

        double s = b;
        for (int i = 0; i < len; i++)
        {
            s += w[i] * ins[i];
        }

        if (std::isnan(s) || std::isinf(s)){
            s = 0.0;
        }
        double current_output = std::tanh(s);

        jfieldID self_pID = env->GetFieldID(cls, "prediction", "D");
        jfieldID self_wID = env->GetFieldID(cls, "selfWeight", "D");

        double last_p = env->GetDoubleField(obj, self_pID);
        double sw = env->GetDoubleField(obj, self_wID);

        double diff = std::abs(current_output - last_p);
        if (diff < SPARTAN_LIMIT)
        {
            sw += (SPARTAN_LIMIT - diff) * PI;
            if (sw > 10.0){
                sw = 10.0;
            }
            env->SetDoubleField(obj, self_wID, sw);
        }

        env->SetDoubleField(obj, self_pID, current_output);
        return current_output;
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_train(JNIEnv *env, jobject obj, jdouble target)
    {

        static bool seeded = false;
        if (!seeded) {
            srand(static_cast<unsigned int>(time(NULL))); 
            seeded = true;
        }

        jclass cls = env->GetObjectClass(obj);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        jdoubleArray iA = (jdoubleArray)env->GetObjectField(obj, env->GetFieldID(cls, "inputs", "[D"));
        std::vector<double> ins(len);
        env->GetDoubleArrayRegion(iA, 0, len, ins.data());

        double lr = env->GetDoubleField(obj, env->GetFieldID(cls, "learningRate", "D"));
        double b = env->GetDoubleField(obj, env->GetFieldID(cls, "bias", "D"));

        jobjectArray peersA = (jobjectArray)env->GetObjectField(obj, env->GetFieldID(cls, "peers", "[Lai/Neuron;"));
        double collectiveError = 0.0;
        if (peersA != NULL)
        {
            jsize peerCount = env->GetArrayLength(peersA);
            double totalPeerOut = 0.0;
            for (int i = 0; i < peerCount; i++)
            {
                jobject peer = env->GetObjectArrayElement(peersA, i);
                jmethodID predID = env->GetMethodID(env->GetObjectClass(peer), "predict", "()D");
                totalPeerOut += env->CallDoubleMethod(peer, predID);
            }
            collectiveError = target - (totalPeerOut / peerCount);
        }

        double myOut = Java_ai_Neuron_predict(env, obj);
        double myError = target - myOut;
        double delta = ((peersA != NULL) ? (myError - collectiveError) : myError) * (1.0 - myOut * myOut);

        std::map<int, double> loyalties;
        for (int i = 0; i < len; i++)
        {
            loyalties[i] = w[i] * PI;
            if (std::abs(loyalties[i]) < SPARTAN_LIMIT)
            {
                w[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }

        int b_id = 0, l_id = 0;
        for (int i = 1; i < len; i++)
        {
            if (w[i] > w[b_id])
                b_id = i;
            if (w[i] < w[l_id])
                l_id = i;
        }

        w[b_id] *= std::abs(loyalties[b_id]);
        w[l_id] *= std::abs(loyalties[l_id]);

        for (int i = 0; i < len; i++)
        {
            double weight_loyalty = std::abs(w[i] * PI);
            w[i] += lr * delta * ins[i] * weight_loyalty;
        }
        b += lr * delta;

        env->SetDoubleArrayRegion(wA, 0, len, w.data());
        env->SetDoubleField(obj, env->GetFieldID(cls, "bias", "D"), b);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_motivate(JNIEnv *env, jobject obj, jdouble reward, jdouble gamma, jdouble state)
    {
        jclass cls = env->GetObjectClass(obj);
        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        double b = env->GetDoubleField(obj, bF);

        jfieldID lrF = env->GetFieldID(cls, "learningRate", "D");
        double lr = env->GetDoubleField(obj, lrF);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL)
        {
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        double out = Java_ai_Neuron_predict(env, obj);
        double trg = reward + gamma * state;
        double err = trg - out;

        double deriv = 1.0 - (out * out);

        for (jsize i = 0; i < len; ++i)
        {
            w[i] += lr * err * deriv;
        }
        b += lr * err * deriv;

        env->SetDoubleArrayRegion(wA, 0, len, w.data());
        env->SetDoubleField(obj, bF, b);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_exportWeights(JNIEnv *env, jobject obj, jstring filename)
    {
        const char *nm = env->GetStringUTFChars(filename, 0);
        jclass cls = env->GetObjectClass(obj);

        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        double b = env->GetDoubleField(obj, bF);

        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL)
        {
            env->ReleaseStringUTFChars(filename, nm);
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);
        env->GetDoubleArrayRegion(wA, 0, len, w.data());

        std::ofstream f(nm);
        for (jsize i = 0; i < len; ++i)
        {
            f << w[i] << " ";
        }
        f << b << std::endl;
        f.close();

        env->ReleaseStringUTFChars(filename, nm);
    }

    JNIEXPORT void JNICALL Java_ai_Neuron_importWeights(JNIEnv *env, jobject obj, jstring filename)
    {
        const char *nm = env->GetStringUTFChars(filename, 0);
        jclass cls = env->GetObjectClass(obj);

        jfieldID bF = env->GetFieldID(cls, "bias", "D");
        jfieldID wF = env->GetFieldID(cls, "weights", "[D");
        jdoubleArray wA = (jdoubleArray)env->GetObjectField(obj, wF);
        if (wA == NULL)
        {
            env->ReleaseStringUTFChars(filename, nm);
            return;
        }

        jsize len = env->GetArrayLength(wA);
        std::vector<double> w(len);

        std::ifstream f(nm);
        if (f.is_open())
        {
            for (jsize i = 0; i < len; ++i)
            {
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