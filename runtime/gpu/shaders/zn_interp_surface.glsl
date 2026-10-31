// Zonaite frame interpolation : what the choose and compose passes share. A surface is what one
// forward warp (zn_interp_scatter) left at a pixel of the in-between frame: which frame it came from, its motion
// to the other frame and its view distance.
#include "zn_interp_depth.glsl"

layout(push_constant) uniform Compose {
    vec2 out_size;
    float t;
    int pad;
    vec4 content;  // the game's picture in the frame (x, y, width, height as fractions); outside it, bars
} p;

layout(binding = 0) uniform sampler2D scene_prev;  // display-encoded
layout(binding = 1) uniform sampler2D scene_cur;

struct Surface {
    bool ok;
    int from;  // 0 = the current frame, 1 = the previous frame
    vec2 v;    // motion from its frame to the other
    float distance;
    bool solo;  // clean mode: a thing moving by itself, from the nearer frame only (never blended)
};

// Where surface s, seen at uv at time t, is in the previous (xy) and the current (zw) frame
vec4 Sources(Surface s, vec2 uv) {
    if (s.from == 0) {
        const vec2 xc = uv - (1.0 - p.t) * s.v;
        return vec4(xc + s.v, xc);
    }
    const vec2 xp = uv - p.t * s.v;
    return vec4(xp, xp + s.v);
}

// A colour sample between texels with a Catmull-Rom filter (9 bilinear taps): as sharp as the frames themselves,
// where bilinear sampling would soften every generated frame a little against the real ones around it
vec3 sharp(sampler2D tex, vec2 uv) {
    const vec2 size = vec2(textureSize(tex, 0));
    const vec2 pos = uv * size;
    const vec2 tc = floor(pos - 0.5) + 0.5;
    const vec2 f = pos - tc;
    const vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    const vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    const vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    const vec2 w3 = f * f * (-0.5 + 0.5 * f);
    const vec2 w12 = w1 + w2;
    const vec2 t0 = (tc - 1.0) / size, t12 = (tc + w2 / w12) / size, t3 = (tc + 2.0) / size;
    vec3 c = (texture(tex, vec2(t0.x, t0.y)).rgb * w0.x + texture(tex, vec2(t12.x, t0.y)).rgb * w12.x +
              texture(tex, vec2(t3.x, t0.y)).rgb * w3.x) * w0.y;
    c += (texture(tex, vec2(t0.x, t12.y)).rgb * w0.x + texture(tex, vec2(t12.x, t12.y)).rgb * w12.x +
          texture(tex, vec2(t3.x, t12.y)).rgb * w3.x) * w12.y;
    c += (texture(tex, vec2(t0.x, t3.y)).rgb * w0.x + texture(tex, vec2(t12.x, t3.y)).rgb * w12.x +
          texture(tex, vec2(t3.x, t3.y)).rgb * w3.x) * w3.y;
    return clamp(c, 0.0, 1.0);
}

// Colour of surface s at uv: from both frames where they agree on it, else from its own frame only
vec3 One(Surface s, vec2 uv) {
    const vec4 at = Sources(s, uv);
    const vec3 a = sharp(scene_prev, at.xy), b = sharp(scene_cur, at.zw);
    const vec3 d = abs(a - b);
    if (!s.solo && d.r + d.g + d.b < 0.1) return mix(a, b, p.t);
    return s.from == 0 ? b : a;
}

// The choose pass's verdict per pixel (rgba32f): the surface's motion and distance, and how to colour it
const float kNone = 0.0;  // nothing landed nearby: blend the frames in place
const float kOne = 1.0;   // one surface (One), from frame `from`: code = kOne + from
const float kBoth = 3.0;  // the same surface from both frames, stored as the current frame's (from 0)
