// One SIMD group owns one 32-channel scale and its signed codes.
uint group = threadgroup_position_in_grid.x;
uint lane = thread_index_in_threadgroup;
float value = float(x[group * 32 + lane]);
float magnitude = simd_max(abs(value));
float scale = magnitude > 0.0f ? max(magnitude / 127.0f, 1.0e-30f) : 1.0f;
codes[group * 32 + lane] = char(clamp(rint(value / scale), -127.0f, 127.0f));
if (lane == 0) scales[group] = scale;
