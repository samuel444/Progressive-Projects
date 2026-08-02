
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

inline double bs_price(
    double spot, double strike, double time, double rate,
    double dividend_yield, double volatility, bool is_call
) {
    if (!(spot > 0.0) || !(strike > 0.0) || !(time >= 0.0) || !(volatility > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (time == 0.0) {
        return is_call ? std::max(spot - strike, 0.0)
                       : std::max(strike - spot, 0.0);
    }
    const double sqrt_t = std::sqrt(time);
    const double d1 = (
        std::log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time
    ) / (volatility * sqrt_t);
    const double d2 = d1 - volatility * sqrt_t;
    const double discounted_spot = spot * std::exp(-dividend_yield * time);
    const double discounted_strike = strike * std::exp(-rate * time);
    if (is_call) {
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2);
    }
    return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1);
}

py::array_t<double> black_scholes_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> spot,
    py::array_t<double, py::array::c_style | py::array::forcecast> strike,
    py::array_t<double, py::array::c_style | py::array::forcecast> time,
    py::array_t<double, py::array::c_style | py::array::forcecast> rate,
    py::array_t<double, py::array::c_style | py::array::forcecast> dividend_yield,
    py::array_t<double, py::array::c_style | py::array::forcecast> volatility,
    bool is_call
) {
    const auto s = spot.unchecked<1>();
    const auto k = strike.unchecked<1>();
    const auto t = time.unchecked<1>();
    const auto r = rate.unchecked<1>();
    const auto q = dividend_yield.unchecked<1>();
    const auto v = volatility.unchecked<1>();
    const py::ssize_t n = s.shape(0);
    if (k.shape(0) != n || t.shape(0) != n || r.shape(0) != n || q.shape(0) != n || v.shape(0) != n) {
        throw std::invalid_argument("All arrays must have equal length");
    }
    py::array_t<double> output(n);
    auto out = output.mutable_unchecked<1>();
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
        out(i) = bs_price(s(i), k(i), t(i), r(i), q(i), v(i), is_call);
    }
    return output;
}

py::array_t<double> implied_volatility_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> market_price,
    py::array_t<double, py::array::c_style | py::array::forcecast> spot,
    py::array_t<double, py::array::c_style | py::array::forcecast> strike,
    py::array_t<double, py::array::c_style | py::array::forcecast> time,
    py::array_t<double, py::array::c_style | py::array::forcecast> rate,
    py::array_t<double, py::array::c_style | py::array::forcecast> dividend_yield,
    bool is_call,
    double lower = 1e-6,
    double upper = 5.0,
    int iterations = 80
) {
    const auto m = market_price.unchecked<1>();
    const auto s = spot.unchecked<1>();
    const auto k = strike.unchecked<1>();
    const auto t = time.unchecked<1>();
    const auto r = rate.unchecked<1>();
    const auto q = dividend_yield.unchecked<1>();
    const py::ssize_t n = m.shape(0);
    py::array_t<double> output(n);
    auto out = output.mutable_unchecked<1>();
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
        double lo = lower, hi = upper;
        const double target = m(i);
        if (!(target >= 0.0) || !(s(i) > 0.0) || !(k(i) > 0.0) || !(t(i) > 0.0)) {
            out(i) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double flo = bs_price(s(i), k(i), t(i), r(i), q(i), lo, is_call) - target;
        const double fhi = bs_price(s(i), k(i), t(i), r(i), q(i), hi, is_call) - target;
        if (flo * fhi > 0.0) {
            out(i) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        for (int j = 0; j < iterations; ++j) {
            const double mid = 0.5 * (lo + hi);
            const double fmid = bs_price(s(i), k(i), t(i), r(i), q(i), mid, is_call) - target;
            if (fmid > 0.0) hi = mid; else lo = mid;
        }
        out(i) = 0.5 * (lo + hi);
    }
    return output;
}

PYBIND11_MODULE(fast_options, m) {
    m.doc() = "Optional batch Black-Scholes and implied-volatility acceleration";
    m.def("black_scholes_batch", &black_scholes_batch);
    m.def("implied_volatility_batch", &implied_volatility_batch,
          py::arg("market_price"), py::arg("spot"), py::arg("strike"),
          py::arg("time"), py::arg("rate"), py::arg("dividend_yield"),
          py::arg("is_call"), py::arg("lower") = 1e-6,
          py::arg("upper") = 5.0, py::arg("iterations") = 80);
}
