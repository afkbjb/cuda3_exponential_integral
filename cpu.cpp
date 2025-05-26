#include "ei.h"
#include <cmath>
#include <cfloat>
#include <limits>

float exponentialIntegralFloat(int order, float x) {
    const float EULER = 0.5772156649015329f;
    const float EPS   = 1e-30f;
    const int MAX_IT  = 2000000000;
    float ans, h, b, c, d, fact, psi;
    int nm1 = order - 1;
    if (order == 0) {
        return expf(-x) / x;
    } else if (x > 1.0f) {
        b = x + order;
        c = FLT_MAX;
        d = 1.0f / b;
        h = d;
        for (int i = 1; i <= MAX_IT; ++i) {
            float a = -i * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            float del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= EPS) break;
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0 ? 1.0f/nm1 : -logf(x) - EULER);
        fact = 1.0f;
        for (int i = 1; i <= MAX_IT; ++i) {
            fact *= -x / i;
            float del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -EULER;
                for (int j = 1; j <= nm1; ++j) psi += 1.0f/j;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * EPS) break;
        }
        return ans;
    }
}

double exponentialIntegralDouble(int order, double x) {
    const double EULER = 0.5772156649015329;
    const double EPS   = 1e-30;
    const int MAX_IT   = 2000000000;
    double ans, h, b, c, d, fact, psi;
    int nm1 = order - 1;
    if (order == 0) {
        return exp(-x) / x;
    } else if (x > 1.0) {
        b = x + order;
        c = std::numeric_limits<double>::max();
        d = 1.0 / b;
        h = d;
        for (int i = 1; i <= MAX_IT; ++i) {
            double a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            double del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= EPS) break;
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0 ? 1.0/nm1 : -log(x) - EULER);
        fact = 1.0;
        for (int i = 1; i <= MAX_IT; ++i) {
            fact *= -x / i;
            double del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -EULER;
                for (int j = 1; j <= nm1; ++j) psi += 1.0/j;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * EPS) break;
        }
        return ans;
    }
}