#version 460

layout(location = 0) out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5f);
    fragColor = vec4(0.1f, 0.25f, 1.0f, 1.0f - 2.0f * length(coord));
}