#pragma once

#include <lwpr.hh> //here to avoid external library the necessity to find lwpr lib
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>

namespace lwpr_wrapper
{

template<int N_IN, int N_OUT, int N_SAMPLES>
struct LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::Impl {
    std::unique_ptr<LWPR_Object> lwpr;

    Impl(int n_in, int n_out)
        : lwpr(std::make_unique<LWPR_Object>(n_in, n_out)) {}
};

template<int N_IN, int N_OUT, int N_SAMPLES>
LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::LWPRWrapper() {}

template<int N_IN, int N_OUT, int N_SAMPLES>
LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::~LWPRWrapper() = default;

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::init() {

    // Default is one
    Eigen::Vector<double, N_IN> expected_in_min;
    Eigen::Vector<double, N_IN> expected_in_max;
    Eigen::Vector<double, N_OUT> expected_out_min;
    Eigen::Vector<double, N_OUT> expected_out_max;

    //IN: x y z sin(yaw) cos(yaw)
    if (N_IN ==2) {
        expected_in_min << -8.0, -8.0;
        expected_in_max << 8.0,  8.0;
    } else if (N_IN == 3) {
        expected_in_min << -8.0, -8.0, 0.0;
        expected_in_max << 8.0,  8.0, 8.0;
    } else if (N_IN == 5) {
        expected_in_min << -8.0, -8.0, 0.0, -1.0, -1.0;
        expected_in_max << 8.0,  8.0, 8.0, 1.0, 1.0;
    }
    expected_out_min << -0.5, -1.3; // joint limits
    expected_out_max << 1.48, 1.3; // joint limits
    predict_cutoff_ = 0.001; //default 0.001
    impl_ = std::make_unique<Impl>(N_IN, N_OUT);

    // The expected range of each input, divided by 2, just be consistent with init_D
    // EG. if inputs are 2D position, and we are exploring a square of 2 by 2 meters, norm_is should be [1,1],
    impl_->lwpr->normIn( (expected_in_max - expected_in_min) / 2.0);
    impl_->lwpr->normOut( (expected_out_max - expected_out_min) / 2.0);

    // Set to False useful for finding right parameters 
    impl_->lwpr->updateD(true);

    Eigen::Vector<double, N_IN> D_diag;
    if (N_IN ==2) {
        D_diag << 10,10;
    } else if (N_IN == 3) {
        D_diag << 10,10,10;
    } else if (N_IN == 5) {
        //With 50, result a yaw as
        // 8deg change in drone yaw produces noticeably different camera corrections
        D_diag << 1,1,1,5,5;
    }
    impl_->lwpr->setInitD(D_diag); // default: 25
    impl_->lwpr->setInitAlpha(250); // default: 50
    impl_->lwpr->wGen(0.3); // default: 0.1
    impl_->lwpr->wPrune(0.8); // default: 0.9
    impl_->lwpr->useMeta(false);

    //impl_->lwpr->initLambda(0.99); //default = 0.999
    //impl_->lwpr->finalLambda(0.999); //default = 0.99999
    //impl_->lwpr->tauLambda(0.9999); //default =  0.9999

    w_conf_thresh_ = 0.6;
    prediction_conf_.setZero();
    prediction_maxW_.setZero();
    prediction_out_.setZero();
    update_out_.setZero();

    rng_ = std::mt19937(std::random_device{}());

    lwpr_info_msg.mean_data.resize(N_IN);
    lwpr_info_msg.var_data.resize(N_IN);
    lwpr_info_msg.prediction.resize(N_OUT);
    lwpr_info_msg.prediction_conf.resize(N_OUT);
    lwpr_info_msg.prediction_maxw.resize(N_OUT);
    lwpr_info_msg.num_rfs.resize(N_OUT);
    lwpr_info_msg.rfs_info.resize(N_OUT);

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::run(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& sample) {

    std::cout << "sample is " << sample.transpose() << std::endl;
    predict(input);
    std::cout << "predicted with maxW=" << prediction_maxW_.transpose() << std::endl;
    
    bool to_update = false;
    if (impl_->lwpr->nData() < 1000) {
        to_update = true;
        std::cout << "WARN: too few samples so far, updating. " << impl_->lwpr->nData() << std::endl;
    } else {
        for (int i=0; i<N_OUT; ++i) {
            if (prediction_maxW_(i) < impl_->lwpr->wGen()) { 
                to_update = true;
                std::cout << "WARN: MaxW too low: " << prediction_maxW_.transpose() <<
                    ", conf: " << prediction_conf_.transpose() << 
                    ", maxw: " << prediction_maxW_.transpose() << 
                    " with input " << input.transpose() << 
                    ". The model will be updated with this sample" << std::endl;
                break;
            }
        }
    }

    if (to_update) {

        // this return useful?
        update_out_ = impl_->lwpr->update(input, sample);
        std::cout << "   Update result out: " << update_out_.transpose() << std::endl;
    }

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::add_sample(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output) {

    if (samples_counter_ == N_SAMPLES) {
        std::cout << "Samples buffer is full, cant add this sample" << std::endl;
        return false;
    }

    samples_in_.col(samples_counter_) = input;
    samples_out_.col(samples_counter_) = output;
    samples_counter_++;

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::delete_all_samples() {
    samples_counter_ = 0;
    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::train_immediately() { 

    if (samples_counter_ == 0) {
        std::cout << "Cannot train, no samples" << std::endl;
        return false;
    }

    auto t_start = std::chrono::steady_clock::now();
    std::vector<int> idx;
    idx.resize(samples_counter_);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng_);

    for (const int i : idx) {
        train_single(samples_in_.col(i), samples_out_.col(i));
    }
    int samples_counter_old = samples_counter_;
    delete_all_samples();

    auto t_end = std::chrono::steady_clock::now();
    auto interval_msec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start);
    std::cout << "Training " << samples_counter_old << " required " 
                << interval_msec.count() << "ms" << std::endl;

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::train_immediately(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output) { 

    if (samples_counter_ < N_SAMPLES) { //accumulate
        add_sample(input, output);
    } 

    train_immediately();

    return true;

}

template<int N_IN, int N_OUT, int N_SAMPLES>
Eigen::VectorXd LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::train_single(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output) { 

    return (impl_->lwpr->update(input, output));
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::train(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output) {

    if (samples_counter_ < N_SAMPLES) { //accumulate
        add_sample(input, output);
    } 

    if (samples_counter_ == N_SAMPLES) {
        train_immediately();
    }

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::predict(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input) {

    //std::cout << "input is " << input.transpose() << std::endl;
    prediction_out_ = impl_->lwpr->predict(input, prediction_conf_, prediction_maxW_, predict_cutoff_);
    //std::cout << "predicted with maxW=" << prediction_maxW_.transpose() << std::endl;

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
Eigen::Vector<double, N_OUT> LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::get_prediction() const {
    return prediction_out_;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
Eigen::Vector<double, N_OUT> LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::get_prediction_conf() const {
    return prediction_conf_;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
Eigen::Vector<double, N_OUT> LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::get_prediction_maxW() const {
    return prediction_maxW_;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
std::array<bool, N_OUT> LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::is_confident_about_out() const {    
    std::array<bool, N_OUT> ret;
    for (size_t i=0; i<N_OUT; i++) {
        ret.at(i) = (prediction_maxW_(i) >= w_conf_thresh_); 
    }
    return ret;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::is_confident() const {    
    for (size_t i=0; i<N_OUT; i++) {
        if (prediction_maxW_(i) < w_conf_thresh_) {
            return false;
        } 
    }
    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
lwpr_wrapper_msg::msg::LWPRInfo LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::getMsgInfo() {
    if (!impl_) {
        std::cerr << "Error: LWPR model not initialized. Call init() first." << std::endl;
        return lwpr_wrapper_msg::msg::LWPRInfo{};  // Return empty message
    }

    lwpr_info_msg.n_data = impl_->lwpr->nData();

    for(size_t i = 0; i < N_IN; i++) {
        lwpr_info_msg.mean_data.at(i) = impl_->lwpr->meanX().at(i);
        lwpr_info_msg.var_data.at(i) = impl_->lwpr->varX().at(i);
    }

    for(int i = 0; i < N_OUT; i++) {
        lwpr_info_msg.prediction.at(i) = prediction_out_(i);
        lwpr_info_msg.prediction_conf.at(i) = prediction_conf_(i);
        lwpr_info_msg.prediction_maxw.at(i) = prediction_maxW_(i);
        lwpr_info_msg.num_rfs.at(i) = impl_->lwpr->numRFS().at(i);
    }

    for (int out=0; out < N_OUT; out++) {
        lwpr_info_msg.rfs_info.at(out).n_data.resize(impl_->lwpr->numRFS().at(out));
        lwpr_info_msg.rfs_info.at(out).mean_data.resize(impl_->lwpr->numRFS().at(out));
        lwpr_info_msg.rfs_info.at(out).var_data.resize(impl_->lwpr->numRFS().at(out));
        lwpr_info_msg.rfs_info.at(out).trustworthy.resize(impl_->lwpr->numRFS().at(out));
        lwpr_info_msg.rfs_info.at(out).w.resize(impl_->lwpr->numRFS().at(out));
        for (int i=0; i<impl_->lwpr->numRFS().at(out); i++) {
            const auto rf = impl_->lwpr->getRF(out, i);
            lwpr_info_msg.rfs_info.at(out).n_data.at(i) = rf.w();
            lwpr_info_msg.rfs_info.at(out).mean_data.at(i) = rf.w();
            lwpr_info_msg.rfs_info.at(out).var_data.at(i) = rf.w();
            lwpr_info_msg.rfs_info.at(out).trustworthy.at(i) = rf.w();
            lwpr_info_msg.rfs_info.at(out).w.at(i) = rf.trustworthy();
        }
    }

    return lwpr_info_msg;
}

}  // namespace lwpr_wrapper
