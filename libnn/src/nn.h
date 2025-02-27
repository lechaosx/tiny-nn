#pragma once

#include <functional>

#include <Eigen/Core>

using Activation           = std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)>;
using ActivationDerivative = std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)>;
using LossDerivative       = std::function<Eigen::MatrixXf(const Eigen::MatrixXf &)>;

struct Coefficients {
	Eigen::MatrixXf weights;
	Eigen::VectorXf biases;
};