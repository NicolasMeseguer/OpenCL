// elementwise_loop
__kernel void elementwise(__global const double *a, 
                          __global const double *b,
                          __global double *out, 
                          ulong stride,
                          ulong vector_length) {

  __private unsigned long idx = (get_local_size(0) * get_group_id(0)) + get_local_id(0);

  for (; idx < vector_length; idx += stride) {
    out[idx] = a[idx] * b[idx];
  }
}