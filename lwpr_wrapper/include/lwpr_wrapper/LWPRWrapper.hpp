#pragma once

#include <memory>
#include <random>
#include <Eigen/Dense>

#include <lwpr_wrapper_msg/msg/lwpr_info.hpp>
//#include <lwpr_wrapper/msg/rfs_info.hpp>

namespace lwpr_wrapper
{

template<int N_IN, int N_OUT, int N_SAMPLES = 100>
class LWPRWrapper
{
public:

    static constexpr int N_IN_c = N_IN;
    static constexpr int N_OUT_c = N_OUT;

    LWPRWrapper();  
    ~LWPRWrapper();

    void set_expected_input_ranges(
        const Eigen::Ref<const Eigen::Vector<double, N_IN>>& expected_in_min,
        const Eigen::Ref<const Eigen::Vector<double, N_IN>>& expected_in_max);
    void set_expected_output_ranges(
        const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& expected_out_min,
        const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& expected_out_max);
    void set_predict_cutoff(const double& alpha);
    void set_update_D(const bool& update);
    void set_initial_D(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& D_diag);
    void set_initial_alpha(const double& alpha);
    void set_w_gen(const double& w_gen);
    void set_w_prune(const double& w_prune);
    void set_use_meta(const bool& meta);
    void set_w_conf_tresh(const double& w_conf_thresh);

    bool add_sample(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output);
    bool delete_all_samples();
    bool predict(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input);
    Eigen::VectorXd train_single(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output);
    bool train_immediately();
    bool train_immediately(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output);
    bool train(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output);
    
    bool run(const Eigen::Ref<const Eigen::Vector<double, N_IN>>& input, 
            const Eigen::Ref<const Eigen::Vector<double, N_OUT>>& output);
   
    Eigen::Vector<double, N_OUT> get_prediction() const;
    Eigen::Vector<double, N_OUT> get_prediction_conf() const;
    Eigen::Vector<double, N_OUT> get_prediction_maxW() const;
    std::array<bool, N_OUT> is_confident_about_out() const;
    bool is_confident() const;
    lwpr_wrapper_msg::msg::LWPRInfo getMsgInfo();

private:
    struct Impl;                          // forward declaration
    std::unique_ptr<Impl> impl_;         // hidden implementation

    double predict_cutoff_;
    double w_conf_thresh_;
    Eigen::Vector<double, N_OUT> prediction_conf_;
    Eigen::Vector<double, N_OUT> prediction_maxW_;
    Eigen::Vector<double, N_OUT> prediction_out_;
    Eigen::Vector<double, N_OUT> update_out_;

    Eigen::Matrix<double, N_IN, N_SAMPLES> samples_in_;
    Eigen::Matrix<double, N_OUT, N_SAMPLES> samples_out_;
    int samples_counter_ = 0;
    std::mt19937 rng_;

    lwpr_wrapper_msg::msg::LWPRInfo lwpr_info_msg;


};

// Type aliases for default template parameters
using LWPRWrapper5x2 = LWPRWrapper<5, 2, 100>;

}  // namespace lwpr_wrapper

// Include the template implementation
#include <lwpr_wrapper/LWPRWrapper.tpp>