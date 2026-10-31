// Zonaite frame interpolation , shared by the passes that use the game's depth and camera.
// Descriptor layout: 0-5 = sampled images (bilinear, clamp to edge), 6-10 = storage images.
// Motion is in UV units of the presented frame throughout; depths are view distances (world units, the camera's
// near plane at 0.1). Two motion maps per pair of game frames: the current frame's pixels towards the previous
// frame ("from current") and the previous frame's towards the current one ("from previous").
layout(local_size_x = 8, local_size_y = 8) in;
#include "zn_interp_color.glsl"

const uint kNoDepth = 0xffffffffu;  // the cleared value of the scatter depth images
