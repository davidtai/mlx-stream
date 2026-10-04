// An MLX-owned array without a fill: one buffer from MLX's allocator (MetalAllocator::malloc), wrapped as an evaluated
// array that frees it through the allocator. No kernel writes it and nothing is wrapped no-copy; its bytes are whatever
// the allocator hands out. Built against MLX 0.32.2's headers: re-verify on every MLX bump.
#include <cstdint>
#include <exception>
#include <string>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"
#include "mlx_alloc_shim.h"

namespace mx = mlx::core;

namespace {
thread_local std::string g_error;
}  // namespace

extern "C" {

const char* dsv41_alloc_last_error(void) {
  return g_error.c_str();
}

int dsv41_alloc_uninit(mlx_array* out, const int* shape, size_t ndim, mlx_dtype dtype) {
  try {
    mx::Shape s(shape, shape + ndim);
    const mx::Dtype dt = mlx_dtype_to_cpp(dtype);
    size_t n = dt.size();
    for (int d : s) {
      if (d < 0) {
        g_error = "negative dimension";
        return -1;
      }
      n *= static_cast<size_t>(d);
    }
    mx::allocator::Buffer buf = mx::allocator::malloc(n);
    if (buf.ptr() == nullptr) {
      g_error = "allocator returned no buffer";
      return -1;
    }
    mlx_array_set_(*out, mx::array(buf, std::move(s), dt));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  } catch (...) {
    g_error = "dsv41_alloc_uninit failed";
    return -1;
  }
}

}  // extern "C"
