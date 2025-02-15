#pragma once

#include <bit>
#include <fstream>
#include <filesystem>
#include <format>

#include <Eigen/Core>

template <typename T>
constexpr T big_to_local_endian(T value) {
	if constexpr (std::endian::native == std::endian::big) {
		return value;
	} else {
		return std::byteswap(value);
	}
}

template <typename T>
constexpr T read_value(std::ifstream &file) {
	T value;
	file.read(reinterpret_cast<char *>(&value), sizeof(value));
	return value;
}

inline constexpr Eigen::MatrixXf read_idx_images(const std::filesystem::path &filename, int32_t &width, int32_t &height) {
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		throw std::runtime_error(std::format("Cannot open file: {}", filename.string()));
	}

	int32_t magic_number = big_to_local_endian(read_value<int32_t>(file));

	if (magic_number != 2051) {
		throw std::runtime_error(std::format("Magic number mismatch: expected 2051, got {}", magic_number));
	}

	int32_t num_images = big_to_local_endian(read_value<int32_t>(file));

	width = big_to_local_endian(read_value<int32_t>(file));
	height = big_to_local_endian(read_value<int32_t>(file));

	Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> images(width * height, num_images);

	file.read(reinterpret_cast<char *>(images.data()), width * height * num_images);

	return images.cast<float>() / 255.0f;
}

inline constexpr Eigen::MatrixXf read_idx_images(const std::filesystem::path &filename) {
	int32_t width;
	int32_t height;
	return read_idx_images(filename, width, height);
}

inline constexpr Eigen::RowVector<uint8_t, Eigen::Dynamic> read_idx_labels(const std::filesystem::path &filename) {
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		throw std::runtime_error(std::format("Cannot open file: {}", filename.string()));
	}

	int32_t magic_number = big_to_local_endian(read_value<int32_t>(file));

	if (magic_number != 2049) {
		throw std::runtime_error(std::format("Magic number mismatch: expected 2049, got {}", magic_number));
	}

	int32_t num_labels = big_to_local_endian(read_value<int32_t>(file));

	Eigen::RowVector<uint8_t, Eigen::Dynamic> labels(num_labels);

	file.read(reinterpret_cast<char *>(labels.data()), num_labels);

	return labels;
}