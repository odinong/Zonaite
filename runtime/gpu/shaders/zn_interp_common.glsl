// Zonaite frame interpolation: shared declarations. The picture-only passes use one descriptor layout:
// 0, 1, 2 = sampled images (bilinear, clamp to edge), 3 = the storage image written.
layout(local_size_x = 8, local_size_y = 8) in;
#include "zn_interp_color.glsl"

layout(push_constant) uniform Params {
    vec2 dst_size;  // size of the image written
    float t;        // generate: the generated frame's time between the previous (0) and the current (1) frame
    int top;        // flow: 1 at the coarsest level (no coarser estimate to start from), 2 at the finest
    int radius;     // flow: search radius in this level's texels
    float lambda;   // flow: cost per texel of departing from the predicted vector
    int mode;       // flow: 1 = one-way (see the flow pass); convert: the source (see the convert pass)
    int pad;
    vec4 content;   // convert: the game's picture in the frame (x, y, width, height as fractions)
} p;

bool outside(ivec2 xy) { return any(greaterThanEqual(xy, ivec2(p.dst_size))); }
