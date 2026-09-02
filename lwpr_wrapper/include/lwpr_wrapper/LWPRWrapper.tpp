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
LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::LWPRWrapper() {

    impl_ = std::make_unique<Impl>(N_IN, N_OUT);

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
    lwpr_info_msg.rfs_info_per_output.resize(N_OUT);

}

template<int N_IN, int N_OUT, int N_SAMPLES>
LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::~LWPRWrapper() = default;

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_expected_input_ranges(
    const Eigen::Ref<const Eigen::Vector<double, N_IN>>& expected_in_min,
    const Eigen::Ref<const Eigen::Vector<double, N_IN>>& expected_in_max)
{
    // Default is one
    impl_->lwpr->normIn( (expected_in_max - expected_in_min) / 2.0);

    for (int i = 0; i < N_IN; ++i) {
        norm_factor_in_.diagonal()(i) = impl_->lwpr->normIn().at(i);
    }
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_expected_output_ranges(
    const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& expected_out_min,
    const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& expected_out_max)
{
    // Default is one
    impl_->lwpr->normOut( (expected_out_max - expected_out_min) / 2.0);

    for (int i = 0; i < N_OUT; ++i) {
        norm_factor_out_.diagonal()(i) = impl_->lwpr->normOut().at(i);
    }
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_predict_cutoff(
    const double& predict_cutoff)
{
    predict_cutoff_ = predict_cutoff; //default 0.001
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_initial_D(
    const Eigen::Ref<const Eigen::Vector<double, N_IN>>& D_diag)
{
    impl_->lwpr->setInitD(D_diag); // default: 25
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_initial_alpha(
    const double& alpha)
{
    impl_->lwpr->setInitAlpha(alpha); // default: 50
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_w_gen(
    const double& w_gen)
{
    impl_->lwpr->wGen(w_gen); // default: 0.1
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_w_prune(
    const double& w_prune)
{
    impl_->lwpr->wPrune(w_prune); // default: 0.9
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_penalty(
    const double& penalty)
{
    impl_->lwpr->penalty(penalty); // default: 1e-6
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_init_lambda(
    const double& init_lambda)
{
    impl_->lwpr->initLambda(init_lambda); // default: 0.999
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_tau_lambda(
    const double& tau_lambda)
{
    impl_->lwpr->tauLambda(tau_lambda); //default =  0.9999
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_final_lambda(
    const double& final_lambda)
{
    impl_->lwpr->finalLambda(final_lambda); // default: 0.99999
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_init_S2(
    const double& init_S2) 
{
    impl_->lwpr->initS2(init_S2); // default: 1e-10
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_update_D(
    const bool& update) 
{
    impl_->lwpr->updateD(update);
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_diag_only(
    const bool& diag_only) 
{
    impl_->lwpr->diagOnly(diag_only); //default true
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_use_meta(
    const bool& meta)
{
    impl_->lwpr->useMeta(meta);
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_meta_rate(
    const double& meta_rate) 
{
    impl_->lwpr->metaRate(meta_rate); // default: 250
}

template<int N_IN, int N_OUT, int N_SAMPLES>
void LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::set_kernel(
    const std::string& kernel)
{
    impl_->lwpr->kernel(kernel.c_str()); //"Gaussian" (default) or "BiSquare"
} 


template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::run(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& sample) {

    //std::cout << "sample is " << sample.transpose() << std::endl;
    predict(input);
    //std::cout << "predicted with maxW=" << prediction_maxW_.transpose() << std::endl;
    
    bool to_update = false;
    if (impl_->lwpr->nData() < 1000) {
        to_update = true;
        std::cerr << "WARN: too few samples so far, updating. " << impl_->lwpr->nData() << std::endl;
    } else {
        for (int i=0; i<N_OUT; ++i) {
            if (prediction_maxW_(i) < impl_->lwpr->wGen()) { 
                to_update = true;
                std::cerr << "WARN: MaxW too low: " << prediction_maxW_.transpose() <<
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
        //std::cout << "   Update result out: " << update_out_.transpose() << std::endl;
    }

    return true;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
bool LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::add_sample(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output) {

    if (samples_counter_ == N_SAMPLES) {
        std::cerr << "Samples buffer is full, cant add this sample" << std::endl;
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
        std::cerr << "Cannot train, no samples" << std::endl;
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
    if (impl_->lwpr->nData() <= 0) {
        return false;
    }
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
        lwpr_info_msg.prediction.at(i) = prediction_out_(i);
        lwpr_info_msg.prediction_conf.at(i) = prediction_conf_(i);
        lwpr_info_msg.prediction_maxw.at(i) = prediction_maxW_(i);
    }

    for (int out=0; out < N_OUT; out++) {
        lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.resize(impl_->lwpr->numRFS().at(out));
        for (int i=0; i<impl_->lwpr->numRFS().at(out); i++) {
            const auto rf = impl_->lwpr->getRF(out, i);
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).weight = rf.w();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).n_reg = rf.nReg();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).num_data = rf.numData();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).mean_data = rf.meanX();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).var_data = rf.varX();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).center = rf.center();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).d.clear();
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).d.reserve(N_IN * N_IN);
            //row major
            for (const auto& row : rf.D()) {
                lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).d.insert(
                    lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).d.end(), 
                    row.begin(), row.end());
            }
            lwpr_info_msg.rfs_info_per_output.at(out).rfs_info.at(i).trustworthy = rf.trustworthy();
        }
    }

    return lwpr_info_msg;
}

template<int N_IN, int N_OUT, int N_SAMPLES>
visualization_msgs::msg::MarkerArray 
LWPRWrapper<N_IN, N_OUT, N_SAMPLES>::getRFMarkers() const requires (N_IN >= 3){

    visualization_msgs::msg::MarkerArray rfs_markers;
    
    for (int out = 0; out < N_OUT; ++out)
    {
        for (int rf_idx = 0; rf_idx < impl_->lwpr->numRFS().at(out); rf_idx++)
        {
            const auto rf = impl_->lwpr->getRF(out, rf_idx);

            visualization_msgs::msg::Marker marker;

            marker.ns = "out_" + std::to_string(out);
            marker.id = rf_idx;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            //----------------------------------
            // center
            //----------------------------------

            auto center = rf.center();

            marker.pose.position.x = center[0] * norm_factor_in_.diagonal()(0);
            marker.pose.position.y = center[1] * norm_factor_in_.diagonal()(1);
            marker.pose.position.z = center[2] * norm_factor_in_.diagonal()(2);

            //----------------------------------
            // D matrix
            //----------------------------------

            Eigen::Matrix<double, 3, 3> D;
            const auto rf_d = rf.D();

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    D(r,c) = rf_d.at(r).at(c);

            auto N3 = norm_factor_in_.diagonal().template head<3>().cwiseInverse().asDiagonal();

            D = N3 * D * N3;

            //----------------------------------
            // eigendecomposition
            //----------------------------------

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(D);

            const Eigen::Vector<double, 3> lambda = eig.eigenvalues();
            Eigen::Matrix<double, 3, 3> Q = eig.eigenvectors();
            if (Q.determinant() < 0.0) {
                Q.col(0) *= -1.0;   // flip one column to restore a proper rotation
            }

            //----------------------------------
            // orientation
            //----------------------------------

            Eigen::Quaterniond q(Q);
            q.normalize();

            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();

            //----------------------------------
            // size
            //----------------------------------

            constexpr double sigma_level = 2.0; //2 sigma: 95% mass of the gaussian

            marker.scale.x =
                2.0 * sigma_level / std::sqrt(lambda(0));

            marker.scale.y =
                2.0 * sigma_level / std::sqrt(lambda(1));

            marker.scale.z =
                2.0 * sigma_level / std::sqrt(lambda(2));

            //----------------------------------
            // color
            //----------------------------------

            if (out == 0) {
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
            } else if (out == 1) {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
            } else if (out == 2){
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
            } else {
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
            }

            marker.color.a = std::clamp(float(rf.w()), 0.2f, 0.8f);

            rfs_markers.markers.push_back(marker);
        }
    }

    return rfs_markers;
}

}  // namespace lwpr_wrapper
