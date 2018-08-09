#include "render.hpp"
#include <atomic>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <mutex>
#include <vector>
#include <immintrin.h>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/partitioner.h"
#include "tbb/task_scheduler_init.h"

struct rgb8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

rgb8_t heat_lut(float x)
{
  if (x >= 1)
    return rgb8_t{0, 0, 0};

  assert(0 <= x && x <= 1);
  constexpr float x0 = 1.f / 4.f;
  constexpr float x1 = 2.f / 4.f;
  constexpr float x2 = 3.f / 4.f;

  if (x < x0)
  {
    const auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgb8_t{0, g, 255};
  }
  else if (x < x1)
  {
    const auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgb8_t{0, 255, b};
  }
  else if (x < x2)
  {
    const auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgb8_t{r, 255, 0};
  }
  else// (x < 1)
  {
    const auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgb8_t{255, b, 0};
  }
}

void render(std::byte* buffer,
            int width,
            int height,
            std::ptrdiff_t stride,
            int n_iterations)
{

  const int h_height = height % 2 == 0 ? height / 2 : height / 2 + 1;
  const int x_scales_size = (width >> 3) + 1; // dooooo
  const float y_scale = 2.0 / (height - 1);
  const __m256 x_scale = _mm256_set1_ps(3.5 / (width - 1));
  const __m256 const_mask = _mm256_set1_ps(-1);
  const __m256 m1 = _mm256_set1_ps(1);
  const __m256 m4 = _mm256_set1_ps(4);
  const __m256 x_sub_scale = _mm256_set1_ps(2.5);
  std::vector<int> hist(n_iterations + 1, 0);
  int pixels [h_height][width + 7];
  __m256 *x_scales = new __m256[x_scales_size];

  x_scales[x_scales_size - 1] = _mm256_setzero_ps();
  for (int i = 0; i < width; i += 8)
  {
    const __m256 mi = _mm256_set_ps(i + 7,
                                    i + 6,
                                    i + 5,
                                    i + 4,
                                    i + 3,
                                    i + 2,
                                    i + 1,
                                    i);
    x_scales[(i >> 3)] = x_scale * mi - x_sub_scale;
  }

  for (int py = 0; py != h_height; ++py)
  {
    const __m256 y0 = _mm256_set1_ps(py * y_scale - 1);

    for (int px = 0; px != x_scales_size; ++px)
    {
      const __m256 x0 = x_scales[px];
      __m256 x = _mm256_setzero_ps();
      __m256 y = _mm256_setzero_ps();
      __m256 xy = _mm256_setzero_ps();
      __m256 sx = _mm256_setzero_ps();
      __m256 sy = _mm256_setzero_ps();
      __m256 cmp = _mm256_setzero_ps();
      __m256 m_iterations = _mm256_setzero_ps();
      __m256 sx_sy = _mm256_setzero_ps();
      int iteration = 0;

      for (; iteration != n_iterations; ++iteration)
      {
        sx = x * x;
        sy = y * y;
        xy = x * y;
        y = xy + xy + y0;
        x = sx - sy + x0;
        sx_sy = sx + sy;
        cmp = sx_sy < m4;
        m_iterations += _mm256_and_ps(cmp, m1);
        if (_mm256_testz_ps(cmp, const_mask))
          break;
      }

      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(&pixels[py][px << 3]),
          _mm256_cvtps_epi32(m_iterations));
    }
  }
  for (int y = 0; y != h_height; ++y)
    for (int x = 0; x != width; ++x)
      ++hist[pixels[y][x]];

  const double total = std::accumulate(hist.begin(), std::prev(hist.end()), 0);

  std::vector<float> hues(n_iterations, 0);
  hues[0] = hist[0];
  for (int i = 1; i != n_iterations; ++i)
    hues[i] = hues[i - 1] + (float) hist[i] / total;

  auto buffer2 = buffer + (height - 1) * stride;
  for (int py = 0; py < h_height; ++py)
  {
    rgb8_t* lineptr = reinterpret_cast<rgb8_t*>(buffer);
    rgb8_t* lineptr2 = reinterpret_cast<rgb8_t*>(buffer2);

    for (int px = 0; px < width; ++px)
    {
      auto tmp = heat_lut(hues[pixels[py][px]]);
      if (pixels[py][px] == n_iterations)
        tmp = rgb8_t{0,0,0};
      lineptr[px] = tmp;
      lineptr2[px] = tmp;
    }
    buffer += stride;
    buffer2 -= stride;
  }
}


void render_mt(std::byte* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations)
{
  const int h_height = height % 2 == 0 ? height / 2 : height / 2 + 1;
  const int x_scales_size = (width >> 3) + 1; // dooooo
  const float y_scale = 2.0 / (height - 1);
  const __m256 x_scale = _mm256_set1_ps(3.5 / (width - 1));
  const __m256 const_mask = _mm256_set1_ps(-1);
  const __m256 m1 = _mm256_set1_ps(1);
  const __m256 m4 = _mm256_set1_ps(4);
  const __m256 x_sub_scale = _mm256_set1_ps(2.5);
  std::vector<int> hist(n_iterations + 1, 0);
  int pixels [h_height][width + 7];
  __m256 *x_scales = new __m256[x_scales_size];

  x_scales[x_scales_size - 1] = _mm256_setzero_ps();
  for (int i = 0; i < width; i += 8)
  {
    const __m256 mi = _mm256_set_ps(i + 7,
                                    i + 6,
                                    i + 5,
                                    i + 4,
                                    i + 3,
                                    i + 2,
                                    i + 1,
                                    i);
    x_scales[(i >> 3)] = x_scale * mi - x_sub_scale;
  }

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, height / 2),
      [&](const tbb::blocked_range<size_t>& r)
      {
        for (auto py = r.begin(); py < r.end(); py++)
        {
          const __m256 y0 = _mm256_set1_ps(py * y_scale - 1);

          for (int px = 0; px  != x_scales_size; ++px)
          {
            const __m256 x0 = x_scales[px];
            __m256 x = _mm256_setzero_ps();
            __m256 y = _mm256_setzero_ps();
            __m256 xy = _mm256_setzero_ps();
            __m256 sx = _mm256_setzero_ps();
            __m256 sy = _mm256_setzero_ps();
            __m256 cmp = _mm256_setzero_ps();
            __m256 m_iterations = _mm256_setzero_ps();
            __m256 sx_sy = _mm256_setzero_ps();
            int iteration = 0;

            for (; iteration != n_iterations; ++iteration)
            {
              sx = x * x;
              sy = y * y;
              xy = x * y;
              y = xy + xy + y0;
              x = sx - sy + x0;
              sx_sy = sx + sy;
              cmp = sx_sy < m4;
              m_iterations += _mm256_and_ps(cmp, m1);
              if (_mm256_testz_ps(cmp, const_mask))
                break;
            }

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&pixels[py][px << 3]),
                _mm256_cvtps_epi32(m_iterations));
          }
        }
        });

  for (int y = 0; y != h_height; ++y)
    for (int x = 0; x != width; ++x)
      ++hist[pixels[y][x]];

  /*const double total = tbb::parallel_reduce(
                      tbb::blocked_range<std::vector<int>::iterator>(
                        hist.begin(), std::prev(hist.end())), 0,
                      [](tbb::blocked_range<std::vector<int>::iterator> const& r, int value) {
                         return std::accumulate(r.begin(), r.end(), value);
                      },
                      std::plus<int>()
  );*/
  //std::cout << "total: " << total << std::endl;
  const double total = std::accumulate(hist.begin(), std::prev(hist.end()), 0);

  std::vector<double> hues(n_iterations, 0);
  hues[0] = hist[0];
  for (int i = 1; i != n_iterations; ++i)
    hues[i] = hues[i - 1] + (float) hist[i] / total;

  auto buffer2 = buffer + (height - 1) * stride;
  for (auto py = 0; py < h_height; ++py)
  {
    rgb8_t* lineptr = reinterpret_cast<rgb8_t*>(buffer);
    rgb8_t* lineptr2 = reinterpret_cast<rgb8_t*>(buffer2);

    for (int px = 0; px < width; ++px)
    {
      auto tmp = heat_lut(hues[pixels[py][px]]);
      if (pixels[py][px] == n_iterations)
        tmp = rgb8_t{0,0,0};
      lineptr[px] = tmp;
      lineptr2[px] = tmp;
    }
    buffer += stride;
    buffer2 -= stride;

  }
}
